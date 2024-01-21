import pandas as pd
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore, SelfCheckNgram, SelfCheckNLI
from methods.LMvsLM.LM_vs_LM import *
import torch
import spacy
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cuda" if torch.cuda.is_available() else "cpu"
        self.selfcheck_mqag = SelfCheckMQAG(device=device)  # set device to 'cuda' if GPU is available
        self.selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
        self.selfcheck_ngram = SelfCheckNgram(n=1)  # n=1 means Unigram, n=2 means Bigram, etc.
        self.selfcheck_nli = SelfCheckNLI(device="cuda" if torch.cuda.is_available() else "cpu")
        self.client = openai.OpenAI()

    def generate_additional_samples(self, query, model_name, num_samples=20):
        samples = []
        if model_name == "gpt3":
            model = "gpt-3.5-turbo-instruct"
            for i in range(num_samples):
                response = self.client.completions.create(
                    model=model,
                    prompt=query,
                    temperature=1,
                    max_tokens=100)
                samples.append(response.choices[0].text)
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
        elif model_name == "chatgpt":
            model = "gpt3.5-turbo-1106"
            for i in range(num_samples):
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": query},
                    ],
                    temperature=1,
                    max_tokens=100)
                samples.append(response.choices[0].message.content)
        else:
            raise ValueError("Invalid model_name")

        return samples

    def make_predictions(self, row, query, model_name=None):
        output_predictions = row.copy(deep=True)
        # LLM's text (e.g. GPT-3 response) to be evaluated at the sentence level  & Split it into sentences
        sentences = [sent.text.strip() for sent in self.nlp(row['generations']).sents]  # spacy sentence tokenization
        additional_samples = self.generate_additional_samples(query, model_name)
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

        return output_predictions

class LMvsLM(Methods):
    def __init__(self):
        super().__init__()


    def detect_hal(self, query, claim):
        examiner = Examiner(claim)
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

        return lawyer_history[-1]['content'], lawyer_history


    def make_predictions(self, row, query, model_name=None):
        output_predictions = row.copy(deep=True)

        all_history = []


        label_content, history = self.detect_hal(query, row['generations'])
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

        return output_predictions

# if __name__ == "__main__":
#     passage = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Nation."
#     method = SelfCheckGPT(passage)
#     method.make_predictions(passage)