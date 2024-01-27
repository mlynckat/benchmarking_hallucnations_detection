import os

import openai
import spacy
import torch
from dotenv import load_dotenv
from perplexity import Perplexity
from methods.selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore, SelfCheckNgram, SelfCheckNLI

from methods.AlignScore.alignscore import AlignScore
from methods.LMvsLM.LM_vs_LM import *
from methods.sac3 import paraphraser
from methods.sac3.consistency_checker import SemanticConsistnecyCheck
from methods.sac3.evaluator import Evaluate
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
print(torch.cuda.is_available())

class Methods:
    def __init__(self):
        pass


    def make_predictions(self, row, query, model_name=None):
        # Add your prediction logic here
        pass

class SelfCheckGPT(Methods):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")
        device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else "cpu"
        self.selfcheck_mqag = SelfCheckMQAG(device=device)  # set device to 'cuda' if GPU is available
        self.selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
        self.selfcheck_ngram = SelfCheckNgram(n=1)  # n=1 means Unigram, n=2 means Bigram, etc.
        self.selfcheck_nli = SelfCheckNLI(device="cuda" if torch.cuda.is_available() else "cpu")
        self.client = openai.OpenAI()
        self.client_prompting = openai.OpenAI()

    def generate_additional_samples(self, query, model_name, num_samples=10):
        samples = []
        cost = 0
        if model_name == "gpt3":
            model = "gpt-3.5-turbo-instruct"
            for i in range(num_samples):
                response = self.client.completions.create(
                    model=model,
                    prompt=query,
                    temperature=1,
                    max_tokens=100)
                samples.append(response.choices[0].text)
                cost += response.usage.total_tokens
        elif model_name == "gpt4":
            model = "gpt-4-1106-preview"
            for i in range(num_samples):
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": query},
                    ],
                    temperature=1,
                    max_tokens=100)
                samples.append(response.choices[0].message.content)
                cost += response.usage.total_tokens
        elif model_name == "chatgpt":
            model = "gpt-3.5-turbo-1106"
            for i in range(num_samples):
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": query},
                    ],
                    temperature=1,
                    max_tokens=100)
                samples.append(response.choices[0].message.content)
                cost += response.usage.total_tokens
        elif model_name == "perplexityAI":
            for i in range(num_samples):
                perplexity = Perplexity()
                answer = perplexity.search(query)
                for a in answer:
                    if a["status"] == "completed":
                        samples.append(a["answer"])
                perplexity.close()
                cost = 0
        elif model_name == "llama":
            for i in range(num_samples):
                client = openai.OpenAI(api_key=TOGETHER_API_KEY,
                                base_url='https://api.together.xyz',
                                )

                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI assistant",
                        },
                        {
                            "role": "user",
                            "content": query,
                        }
                    ],
                    model="meta-llama/Llama-2-70b-chat-hf",
                    max_tokens=100,
                    temperature=1,
                )
                samples.append(chat_completion.choices[0].message.content)
                cost += chat_completion.usage.total_tokens
        else:
            raise ValueError("Invalid model_name")

        return samples, cost

    def prompting(self, sentences, samples):
        predictions = []
        template = """
                    Context: {context}
                    Sentence: {sentence}
                    Is the sentence supported by the context above?
                    Answer Yes or No:"""
        cost = 0
        for sent in sentences:
            predictions_per_sentence = []

            for sample in samples[0:3]:

                response = self.client_prompting.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {"role": "user", "content": template.format(context=sample, sentence=sent)},
                    ],
                    temperature=0)
                cost += response.usage.total_tokens
                decision = response.choices[0].message.content.split()[0]
                if decision.lower() == "yes":
                    predictions_per_sentence.append(0)
                elif decision.lower() == "no":
                    predictions_per_sentence.append(1)
                else:
                    predictions_per_sentence.append(0.5)
                    logging.info(f"For sentence {sent} the response is {response.choices[0].message.content}")
            averaged_per_sentence = sum(predictions_per_sentence) / len(predictions_per_sentence)
            predictions.append(averaged_per_sentence)

        return predictions, cost

    def make_predictions(self, row, query, model_name=None):
        logging.info(f"Query: {query}")
        logging.info(f"Model name: {model_name}")
        logging.info(f"Generations: {row['generations']}")
        output_predictions = row.copy(deep=True)
        # LLM's text (e.g. GPT-3 response) to be evaluated at the sentence level  & Split it into sentences
        sentences = [sent.text.strip() for sent in self.nlp(row['generations']).sents]  # spacy sentence tokenization
        additional_samples, costs = self.generate_additional_samples(query, model_name)
        logging.info(f"Additional samples: {additional_samples}")
        output_predictions[f'additional_samples_{model_name}'] = additional_samples
        # SelfCheck-MQAG: Score for each sentence where value is in [0.0, 1.0] and high value means non-factual
        # Additional params for each scoring_method:
        # -> counting: AT (answerability threshold, i.e. questions with answerability_score < AT are rejected)
        # -> bayes: AT, beta1, beta2
        # -> bayes_with_alpha: beta1, beta2
        sent_scores_mqag = self.selfcheck_mqag.predict(
            sentences = sentences,               # list of sentences
            passage = row['generations'],                   # passage (before sentence-split)
            sampled_passages = additional_samples, # list of sampled passages
            num_questions_per_sent = 5,          # number of questions to be drawn
            scoring_method = 'bayes_with_alpha', # options = 'counting', 'bayes', 'bayes_with_alpha'
            beta1 = 0.8, beta2 = 0.8,            # additional params depending on scoring_method
        )
        output_predictions['SefCheckGPT_sent_mqag'] = sent_scores_mqag
        output_predictions['SefCheckGPT_mqag'] = sum(sent_scores_mqag)/len(sent_scores_mqag)

        # --------------------------------------------------------------------------------------------------------------- #
        # SelfCheck-BERTScore: Score for each sentence where value is in [0.0, 1.0] and high value means non-factual
        sent_scores_bertscore = self.selfcheck_bertscore.predict(
            sentences = sentences,                          # list of sentences
            sampled_passages = additional_samples, # list of sampled passages
        )
        output_predictions['SefCheckGPT_sent_bertscore'] = sent_scores_bertscore
        output_predictions['SefCheckGPT_bertscore'] = sum(sent_scores_bertscore)/len(sent_scores_bertscore)

        # --------------------------------------------------------------------------------------------------------------- #
        # SelfCheck-Ngram: Score at sentence- and document-level where value is in [0.0, +inf) and high value means non-factual
        # as opposed to SelfCheck-MQAG and SelfCheck-BERTScore, SelfCheck-Ngram's score is not bounded
        sent_scores_ngram = self.selfcheck_ngram.predict(
            sentences=sentences,
            passage=row['generations'],
            sampled_passages=additional_samples,
        )
        # --------------------------------------------------------------------------------------------------------------- #
        output_predictions['SefCheckGPT_sent_ngram'] = sent_scores_ngram['sent_level']['avg_neg_logprob']
        output_predictions['SefCheckGPT_ngram'] = sent_scores_ngram['doc_level']['avg_neg_logprob']

        sent_scores_nli = self.selfcheck_nli.predict(
            sentences=sentences,  # list of sentences
            sampled_passages=additional_samples,  # list of sampled passages
        )
        output_predictions['SefCheckGPT_sent_nli'] = sent_scores_nli
        output_predictions['SefCheckGPT_nli'] = sum(sent_scores_nli) / len(sent_scores_nli)

        # --------------------------------------------------------------------------------------------------------------- #

        sent_scores_prompting, costs_prompting = self.prompting(sentences, additional_samples)
        output_predictions['SefCheckGPT_sent_prompting'] = sent_scores_prompting
        output_predictions['SefCheckGPT_prompting'] = sum(sent_scores_prompting) / len(sent_scores_prompting)
        output_predictions['additional_samples_costs'] = costs
        output_predictions['prompting_costs'] = costs_prompting

        return output_predictions

class LMvsLM(Methods):
    def __init__(self):
        super().__init__()


    def detect_hal(self, query, claim, task):
        examiner = Examiner(claim, task)
        examinee = Suspect(query, claim)
        question = examiner.Setup()
        trigger = True
        count = 1
        while (trigger):
            count += 1
            answer = examinee.answer_without_history(question)
            flag = examiner.check_follow_up_question(answer)
            if 'No' in flag or count == 5:
                lawyer_history = examiner.decision()
                trigger = False
            else:
                question = examiner.ask_continue()
        costs = examinee.cost + examiner.cost

        return lawyer_history[-1]['content'], lawyer_history, costs


    def make_predictions(self, row, query, model_name=None, task=None):
        logging.info(f"Query: {query}")
        logging.info(f"Generations: {row['generations']}")
        logging.info(f"Task: {task}")

        output_predictions = row.copy(deep=True)

        all_history = []


        label_content, history, costs = self.detect_hal(query, row['generations'], task)
        if 'correct' in label_content.lower() and 'incorrect' not in label_content.lower():
            label = 'factual'
        elif 'incorrect' in label_content.lower():
            label = 'non-factual'
        else:
            label = 'non-factual'
        print(label)
        all_history.append(history)

        output_predictions['LMvsLM_label'] = label
        output_predictions['LMvsLM_history'] = all_history
        output_predictions['LMvsLM_costs'] = costs

        return output_predictions

class SAC3(Methods):
    def __init__(self):
        super().__init__()


    def make_predictions(self, row, query, model_name=None, task=None):
        output_predictions = row.copy(deep=True)

        # input information
        question = query
        target_answer = row['generations']

        logging.info(f"Query: {query}")
        logging.info(f"Generations: {row['generations']}")

        # question perturbation
        gen_question, cost_gen_question = paraphraser.paraphrase(question, number=3, model='gpt-3.5-turbo', temperature=1.0)

        # llm evaluation
        llm_evaluate = Evaluate(model='gpt-3.5-turbo')
        self_responses, cost_self_responses = llm_evaluate.self_evaluate(self_question=question, temperature=1.0, self_num=3)
        perb_responses, cost_perb_responses = llm_evaluate.perb_evaluate(perb_questions=gen_question, temperature=0.0)

        logging.info(f"Self responses: {self_responses}")
        logging.info(f"Perb responses: {perb_responses}")

        # consistency check
        scc = SemanticConsistnecyCheck(model='gpt-3.5-turbo')

        sc2_score, sc2_vote, cost_scoring_sc2 = scc.score_scc(question, target_answer, candidate_answers=self_responses, temperature=0.0)
        print(sc2_score, sc2_vote)

        sac3_q_score, sac3_q_vote, cost_scoring_sac3 = scc.score_scc(question, target_answer, candidate_answers=perb_responses,
                                                  temperature=0.0)
        print(sac3_q_score, sac3_q_vote)

        # llm SAC3 QM evaluation
        llm_evaluate = Evaluate(model='falcon-7b')
        falcon_responses, falcon_cost = llm_evaluate.self_evaluate(self_question=question, temperature=1.0, self_num=3)
        falcon_perb_responses, falcon_cost = llm_evaluate.perb_evaluate(perb_questions=gen_question, temperature=0.0)

        all_resp_falcon = falcon_responses + falcon_perb_responses
        logging.info(f"Falcon responses: {all_resp_falcon}")

        scc = SemanticConsistnecyCheck(model='gpt-3.5-turbo')

        sac3_qm_falcon_score, sac3_qm_falcon_vote, cost_scoring_sac3_falcon = scc.score_scc(question, target_answer, candidate_answers=all_resp_falcon,
                                                  temperature=0.0)

        # llm SAC3 QM evaluation
        llm_evaluate = Evaluate(model='starling-7b')
        starling_responses, cost = llm_evaluate.self_evaluate(self_question=question, temperature=1.0, self_num=3)
        starling_perb_responses, cost = llm_evaluate.perb_evaluate(perb_questions=gen_question, temperature=0.0)

        all_resp_starling = starling_responses + starling_perb_responses

        logging.info(f"Starling responses: {all_resp_starling}")

        scc = SemanticConsistnecyCheck(model='gpt-3.5-turbo')

        sac3_qm_starling_score, sac3_qm_starling_vote, cost_scoring_sac3_starling = scc.score_scc(question, target_answer, candidate_answers=all_resp_starling,
                                                    temperature=0.0)

        output_predictions['sc2_score'] = sc2_score
        output_predictions['sc2_vote'] = sc2_vote
        output_predictions['sac3_q_score'] = sac3_q_score
        output_predictions['sac3_q_vote'] = sac3_q_vote
        output_predictions['sac3_qm(falcon)_score'] = sac3_qm_falcon_score
        output_predictions['sac3_qm(falcon)_vote'] = sac3_qm_falcon_vote
        output_predictions['sac3_qm(starling)_score'] = sac3_qm_starling_score
        output_predictions['sac3_qm(starling)_vote'] = sac3_qm_starling_vote
        output_predictions['sc2_cost'] = cost_gen_question + cost_self_responses + cost_scoring_sc2
        output_predictions['sac3_q_cost'] = cost_gen_question + cost_perb_responses + cost_scoring_sac3
        output_predictions['sac3_qm(falcon)_cost'] = cost_gen_question + cost_scoring_sac3_falcon
        output_predictions['sac3_qm(starling)_cost'] = cost_gen_question + cost_scoring_sac3_starling


        return output_predictions

class AlignScorer(Methods):
    def __init__(self):
        super().__init__()

    def make_predictions(self, row, query, model_name=None):
        logging.info(f"Query: {query}")
        logging.info(f"Generations: {row['generations']}")
        logging.info(f"References: {row['references']}")

        output_predictions = row.copy(deep=True)

        scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='methods/AlignScore/AlignScore-base.ckpt',
                            evaluation_mode='nli_sp')
        score = scorer.score(contexts=[row['references']], claims=['generations'])

        output_predictions['AlignScore-base'] = score[0]

        scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0', ckpt_path='methods/AlignScore/AlignScore-large.ckpt',
                            evaluation_mode='nli_sp')
        score = scorer.score(contexts=[row['references']], claims=['generations'])

        output_predictions['AlignScore-large'] = score[0]

        return output_predictions

# if __name__ == "__main__":
#     passage = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Nation."
#     method = SelfCheckGPT(passage)
#     method.make_predictions(passage)