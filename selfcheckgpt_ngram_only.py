import ast
import json
import os

import numpy as np
import pandas as pd
import spacy

from methods.selfcheckgpt.modeling_selfcheck import SelfCheckNgram, SelfCheckBERTScore, SelfCheckMQAG
from methods_implementations import SelfCheckGPT

nlp = spacy.load("en_core_web_sm")
# load config.json
with open("config.json") as json_file:
    config = json.load(json_file)

def apply_ngram(row, model):

    output_predictions = row.copy(deep=True)
    if ("SefCheckGPT_max_ngram" not in row.index) or (row["SefCheckGPT_max_ngram"] != row["SefCheckGPT_max_ngram"]):
        print(f"Working on {row['query']}")
        sentences = [sent.text.strip() for sent in nlp(row['generations'].strip()).sents]
        additional_samples = row[f"additional_samples_{model}"]
        #additional_samples = [sample.replace("\n", " ") for sample in additional_samples]

        sent_scores_ngram = selfcheck_ngram.predict(
            sentences=sentences,
            passage=row['generations'],
            sampled_passages=additional_samples,
        )
        # --------------------------------------------------------------------------------------------------------------- #

        output_predictions['SefCheckGPT_max_ngram'] = sent_scores_ngram['doc_level']['avg_max_neg_logprob']

    return output_predictions

#selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
def apply_berstscore(row):

    output_predictions = row.copy(deep=True)
    sentences = [sent.text.strip() for sent in nlp(row['generations'].strip()).sents]
    additional_samples = row["correct_answer"]
    #additional_samples = [sample.replace("\n", " ") for sample in additional_samples]

    sent_scores_bertscore = selfcheck_bertscore.predict(
        sentences=sentences,  # list of sentences
        sampled_passages=additional_samples,  # list of sampled passages
    )
    output_predictions['SefCheckGPT_bertscore_original'] = sum(sent_scores_bertscore) / len(sent_scores_bertscore)
    # --------------------------------------------------------------------------------------------------------------- #

    return output_predictions

#selfcheck_mqag = SelfCheckMQAG(device="cpu")

def apply_qa(row):

    output_predictions = row.copy(deep=True)
    print(row["SefCheckGPT_mqag_10samples"])
    if row["SefCheckGPT_mqag_10samples"] != row["SefCheckGPT_mqag_10samples"]:
        print(f"Working on {row['query']}")
        sentences = [sent.text.strip() for sent in nlp(row['generations'].strip()).sents]
        additional_samples = row["additional_samples_gpt3"][:10]
        #additional_samples = [sample.replace("\n", " ") for sample in additional_samples]

        sent_scores_mqag = selfcheck_mqag.predict(
            sentences=sentences,  # list of sentences
            passage=row['generations'],  # passage (before sentence-split)
            sampled_passages=additional_samples,  # list of sampled passages
            num_questions_per_sent=5,  # number of questions to be drawn
            scoring_method='bayes_with_alpha',  # options = 'counting', 'bayes', 'bayes_with_alpha'
            beta1=0.8, beta2=0.8,  # additional params depending on scoring_method
        )
        sent_scores_mqag = [x for x in sent_scores_mqag if not np.isnan(x)]
        output_predictions['SefCheckGPT_mqag_10samples'] = np.mean(sent_scores_mqag)
        # --------------------------------------------------------------------------------------------------------------- #

        return output_predictions
    else:
        return output_predictions

# read pandas dataframe from csv
for dataset, model in zip(["FactScore_InstructGPT"], #"PHD_wiki_1y", "PHD_wiki_10w", "PHD_wiki_1000w", "FactScore_PerplexityAI", , "FactScore_ChatGPT", "FAVA_chatgpt", "FAVA_llama",  "FELM_math", "FELM_reasoning", "FELM_science", "FELM_wk", "FELM_writing_rec"
                          ["gpt3"]): #"chatgpt", "chatgpt", "chatgpt", "perplexityAI", , "chatgpt", "chatgpt", "llama", "chatgpt", "chatgpt", "chatgpt", "chatgpt", "chatgpt"
    print(dataset, model)
    path_to_df = f"outputs/{dataset}/SelfCheckGPT_updated_data.csv"

    df = pd.read_csv(path_to_df)
    print(f"The size of the dataset is {df.shape[0]}")
    # drop duplicates
    #df = df.drop(['0.1', "SefCheckGPT_ngram_new", "SefCheckGPT_max_ngram"], axis=1)
    #df = df.dropna(subset=['query', "additional_samples_gpt3", "SefCheckGPT_mqag"])
    df = df.dropna(subset=['query', "generations"])
    df = df.drop_duplicates(subset=['query'], keep='first')
    print(f"The size of the dataset is {df.shape[0]}")
    # drop rows with nan values

    print(f"The size of the dataset is {df.shape[0]}")
    df[f"additional_samples_{model}"] = df[f"additional_samples_{model}"].apply(ast.literal_eval)
    #df['correct_answer'] = df['correct_answer'].apply(ast.literal_eval)
    selfcheck_ngram = SelfCheckNgram(n=1)
    updated_data = df.apply(lambda row: apply_ngram(row, model), axis=1)


    updated_data.to_csv(path_to_df, encoding="utf-8", index=False)
