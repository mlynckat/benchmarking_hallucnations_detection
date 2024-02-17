import ast
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
from jsonlines import jsonlines
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def create_empty_jsonl_file(file_path):
    # Create an empty JSONL file
    with open(file_path, mode='w') as file:
        pass

def add_entry_to_jsonl(file_path, new_entry):
    # Append a new entry to the existing JSONL file
    with jsonlines.open(file_path, mode='a') as writer:
        writer.write(new_entry)

# Set seaborn style
sns.set_style("white")

def transform_prob(label):
    return int(label > 0.5)

def invert_prob(prob):
    return 1 - prob

def transform_factual_to_int(label):
    if label=="factual":
        return 0
    elif label=="non-factual":
        return 1
    else:
        return label

def transform_PHD_labels(label):
    if "False" in str(label):
        return 1
    elif "True" in str(label):
        return 0
    else:
        return label

def calculate_average_label(label):
    # parse as list of lists
    if isinstance(label, str) and "[" in label:
        input_list = ast.literal_eval(label)
        label_mapping = {"accurate": 0, "minor_inaccurate": 0.5, "major_inaccurate": 1}
        substituted_list = [label_mapping[label] for label in input_list]
        average_value = sum(substituted_list) / len(substituted_list)
        return average_value
    else:
        return label

def transform_labels_FactScore(annotations):
    if annotations == 0.0 or annotations == 1.0 or annotations == "0.0" or annotations == "1.0":
        return float(annotations)
    else:
        print(annotations)
        labels = []
        if annotations == annotations:
            annotations = ast.literal_eval(annotations)
            for annotation in annotations:
                if "human-atomic-facts" in annotation:
                    human_atomic_facts = annotation["human-atomic-facts"]
                    if human_atomic_facts == human_atomic_facts:
                        for fact in human_atomic_facts:
                            if "label" in fact:
                                labels.append(fact["label"])

            # Check if there is at least one element other than "S" and "NS" in the list
            invalid_labels = set(labels) - {"S", "NS", "IR"}
            if invalid_labels:
                raise ValueError(f"Invalid label(s) found: {', '.join(invalid_labels)}")

            # Check if there is at least one "NS" in the list of labels
            if "NS" in labels:
                return 1
            else:
                return 0
        else:
            return None

def transform_labels_FELM(label):
    if label == 0 or label == 1:
        return label
    if label:
        if "False" in label:
            return 1
        else:
            return 0
    else:
        return None

def transform_labels_FAVA(label):
    if label == 0 or label == 1:
        return label

    else:
        # Define the pattern using regular expression to capture any text between < and >
        pattern = r'<([^<>]+)>(.*?)<\/\1>'
        # Search for the pattern in the input string
        match = re.search(pattern, label)
        # If a match is found, return 1; otherwise, return 0
        return 1 if match else 0

def transform_labels_BAMBOO(label):
    if label == 0 or label == 1:
        return label
    else:
        if label == False:
            return 1
        elif label == True:
            return 0
        else:
            return None

def transform_labels_ScreenEval(label):
    if label == 0 or label == 1:
        return label
    else:
        if label:
            return 0
        else:
            return 1

def transform_labels_HaluEval(label):
    if label == 0 or label == 1:
        return label
    else:
        if label == "yes":
            return 1
        elif label == "no":
            return 0
        else:
            return None

def return_original(prediction):
    return prediction

config = {
    "SelfCheckGPT": {"columns": ["SefCheckGPT_mqag", "SefCheckGPT_bertscore", "SefCheckGPT_max_ngram", "SefCheckGPT_nli", "SefCheckGPT_prompting"],
                     "transformation": return_original, "thresholds": {"SefCheckGPT_mqag": 0.5, "SefCheckGPT_bertscore": 0.5, "SefCheckGPT_max_ngram": 0.5, "SefCheckGPT_nli": 0.5, "SefCheckGPT_prompting": 0.5}},
    "SelfCheckGPT_10samples": {"columns": ["SefCheckGPT_mqag_10samples", "SefCheckGPT_bertscore_10samples",	"SefCheckGPT_ngram_10samples",	"SefCheckGPT_max_ngram_10_samples",	"SefCheckGPT_nli_10samples"],
                     "transformation": return_original},
    "SelfCheckGPT_alternative": {"columns": ["SefCheckGPT_mqag", "SefCheckGPT_bertscore", "SefCheckGPT_max_ngram", "SefCheckGPT_nli", "SefCheckGPT_prompting"],
                        "transformation": return_original, "thresholds": {"SefCheckGPT_mqag": 0.5, "SefCheckGPT_bertscore": 0.5, "SefCheckGPT_max_ngram": 0.5, "SefCheckGPT_nli": 0.5, "SefCheckGPT_prompting": 0.5}},
    "LMvsLM": {"columns": ["LMvsLM_label"],
               "transformation": transform_factual_to_int, "thresholds": {"LMvsLM_label": 0.5}},
    "SAC3": {"columns": ["sc2_score", "sac3_q_score", "sac3_qm(falcon)_score", "sac3_qm(starling)_score"],
             "transformation": return_original, "thresholds": {"sc2_score": 0.5, "sac3_q_score": 0.5, "sac3_qm(falcon)_score": 0.5, "sac3_qm(starling)_score": 0.5}},
    "AlignScorer": {"columns": ["AlignScore-base", "AlignScore-large"],
                    "transformation": invert_prob, "thresholds": {"AlignScore-base": 0.5, "AlignScore-large": 0.5}},
    "ScaleScorer": {"columns": ["ScaleScorer-large", "ScaleScorer-xl"],
                    "transformation": invert_prob, "thresholds": {"ScaleScorer-large": 0.5, "ScaleScorer-xl": 0.5}},
}

config_benchmarks = {
"SelfCheckGPT": {
    "reference_binary": False,
    "transformation_first": calculate_average_label,
    "transformation_labels": transform_prob
},
"SelfCheckGPT_alternative": {
    "reference_binary": True,
    "transformation_first": return_original,
    "transformation_labels": ""
},
"PHD": {
    "reference_binary": True,
    "transformation_first": transform_PHD_labels,
    "transformation_labels": ""
},
    "FactScore": {
        "reference_binary": True,
        "transformation_first": transform_labels_FactScore,
        "transformation_labels": ""
},
    "FELM": {
        "reference_binary": True,
        "transformation_first": transform_labels_FELM,
        "transformation_labels": ""
},
    "FAVA": {
        "reference_binary": True,
        "transformation_first": transform_labels_FAVA,
        "transformation_labels": ""
},
    "BAMBOO": {
        "reference_binary": True,
        "transformation_first": transform_labels_BAMBOO,
        "transformation_labels": ""
},
    "ScreenEval": {
        "reference_binary": True,
        "transformation_first": transform_labels_ScreenEval,
        "transformation_labels": ""
},
    "HaluEval": {
        "reference_binary": True,
        "transformation_first": transform_labels_HaluEval,
        "transformation_labels": ""
}}

dataset_names_phd = {
    "PHD_wiki_1y": "PHD (High)",
    "PHD_wiki_10w": "PHD (Low)",
    "PHD_wiki_1000w": "PHD (Medium)"
}

def get_data(path_to_df, method, benchmark):
    df = pd.read_csv(path_to_df)
    #print(f"The size of the dataset is {df.shape[0]}")
    # drop duplicates
    df["labels"] = df["labels"].apply(config_benchmarks[benchmark]["transformation_first"])
    na_free = df.dropna(subset=config[method]["columns"]+["labels"])
    #print(f"The size of the dataset after dropping NA's is {na_free.shape[0]}")
    #print(f"Dropped: {df[~df.index.isin(na_free.index)]['generations']}")
    duplicates_free = na_free.drop_duplicates(subset=['generations'], keep='first')
    print(f"The size of the dataset is after removing duplicates: {duplicates_free.shape[0]}")
    #print(f"Dropped: {na_free[~na_free.index.isin(duplicates_free.index)]['generations']}")
    return duplicates_free

def find_optimal_threshold(y_true, y_prob):
    thresholds = np.linspace(0, 1, 100)
    best_threshold = None
    best_score = -float('inf')

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold

def get_optimal_thresholds(labels, predictions, method, benchmark):

    reference_binary_int = config_benchmarks[benchmark]["reference_binary"]
    if not reference_binary_int:
        labels = labels.apply(config_benchmarks[benchmark]["transformation_labels"])

    # print number of labels==1
    print(f"Number of samples with labels==1: {labels[labels == 1].shape[0]}")
    predictions = predictions.apply(config[method]["transformation"])
    # compare to threshold and transform to binary

    best_threshold = find_optimal_threshold(labels, predictions)
    best_predictions = predictions.apply(lambda x: int(x > best_threshold))

    # Calculate evaluation metrics
    accuracy = accuracy_score(labels, best_predictions)
    precision = precision_score(labels, best_predictions)
    recall = recall_score(labels, best_predictions)
    f1_score_val = f1_score(labels, best_predictions)
    # print classification report
    # print(f"Classification report for {method}:")
    # print(classification_report(labels, predictions))

    return accuracy, precision, recall, f1_score_val, best_threshold

create_empty_jsonl_file("outputs/results.jsonl")

for score in ["SelfCheckGPT"]:
    entry = {
        score: {}
    }
    for dataset in ["SelfCheckGPT", "SelfCheckGPT_alternative", "PHD_wiki_1y", "PHD_wiki_10w", "PHD_wiki_1000w", "FactScore_PerplexityAI", "FactScore_InstructGPT", "FactScore_ChatGPT", "FAVA_chatgpt", "FAVA_llama",  "FELM_math", "FELM_reasoning", "FELM_science", "FELM_wk", "FELM_writing_rec"]: # "BAMBOO_abshallu_4k", "BAMBOO_abshallu_16k", "BAMBOO_senhallu_4k", "BAMBOO_senhallu_4k", "ScreenEval_longformer", "ScreenEval_gpt4", "ScreenEval_human", "HaluEval_summarization_data", "HaluEval_dialogue_data", "HaluEval_qa_data" "FAVA_chatgpt", "FAVA_llama",  "FELM_math", "FELM_reasoning", "FELM_science", "FELM_wk", "FELM_writing_rec"
        entry[score][dataset] = {}
        if score == "SelfCheckGPT" and "SelfCheckGPT" in dataset:
            path_to_df = os.path.join("outputs", dataset, f"{score}_updated_data_ngram.csv")
        else:
            path_to_df = os.path.join("outputs", dataset, f"{score}_updated_data.csv")
        if "PHD" in dataset:
            print(f"Dataset name: {dataset_names_phd[dataset]}.")
        else:
            print(f"Dataset name: {dataset}.")
        method = score
        benchmark = dataset.split("/")[-1].split("_")[0]
        df = get_data(path_to_df, method, benchmark)
        for col in config[method]["columns"]:
            accuracy, precision, recall, f1_score_val, best_threshold = get_optimal_thresholds(df["labels"], df[col], method, benchmark)
            entry[score][dataset][col] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score_val, "best_threshold": best_threshold}
            # save to file

add_entry_to_jsonl("outputs/results.jsonl", entry)