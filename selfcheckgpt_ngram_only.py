import ast
import json
import os

import pandas as pd
import spacy

from methods.selfcheckgpt.modeling_selfcheck import SelfCheckNgram
from methods_implementations import SelfCheckGPT

nlp = spacy.load("en_core_web_sm")
# read pandas dataframe from csv

path_to_df = "outputs/SelfCheckGPT/SelfCheckGPT_updated_data.csv"

df = pd.read_csv(path_to_df)
print(f"The size of the dataset is {df.shape[0]}")
# drop duplicates
df = df.drop(['0'], axis=1)
df = df.dropna(subset=['query', "additional_samples_gpt3"])
df = df.drop_duplicates(subset=['query'], keep='first')
print(f"The size of the dataset is {df.shape[0]}")
# drop rows with nan values

print(f"The size of the dataset is {df.shape[0]}")
df['additional_samples_gpt3'] = df['additional_samples_gpt3'].apply(ast.literal_eval)

# load config.json
with open("config.json") as json_file:
    config = json.load(json_file)

selfcheck_ngram = SelfCheckNgram(n=1)

def apply_method(row):

    output_predictions = row.copy(deep=True)
    sentences = [sent.text.strip() for sent in nlp(row['generations'].strip()).sents]
    additional_samples = row["additional_samples_gpt3"]
    additional_samples = [sample.replace("\n", " ") for sample in additional_samples]

    sent_scores_ngram = selfcheck_ngram.predict(
        sentences=sentences,
        passage=row['generations'],
        sampled_passages=additional_samples,
    )
    # --------------------------------------------------------------------------------------------------------------- #

    output_predictions['SefCheckGPT_ngram_new'] = sent_scores_ngram['doc_level']['avg_neg_logprob']
    output_predictions['SefCheckGPT_max_ngram'] = sent_scores_ngram['doc_level']['avg_max_neg_logprob']

    return output_predictions





updated_data = df.apply(lambda row: apply_method(row), axis=1)


updated_data.to_csv(os.path.join("outputs/SelfCheckGPT", "SelfCheckGPT_updated_data_ngram.csv"),
                                    encoding="utf-8", index=False)
