import ast
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
        return int(label)

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
    if label == 0 or label == 1 or label == "0" or label == "1":
        return int(label)
    if label:
        if "False" in label:
            return 1
        else:
            return 0
    else:
        return None

def transform_labels_FAVA(label):
    if label == 0 or label == 1 or label == "0" or label == "1":
        return int(label)

    else:
        # Define the pattern using regular expression to capture any text between < and >
        pattern = r'<([^<>]+)>(.*?)<\/\1>'
        # Search for the pattern in the input string
        match = re.search(pattern, label)
        # If a match is found, return 1; otherwise, return 0
        print(label, 1 if match else 0)
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
    if type(label) == int:
        if label == 0 or label == 1:
            return label
    elif type(label) == bool:
        if label:
            return 0
        else:
            return 1
    else:
        return label

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
                     "transformation": return_original},
    "SelfCheckGPT_10samples": {"columns": ["SefCheckGPT_mqag_10samples", "SefCheckGPT_bertscore_10samples",	"SefCheckGPT_ngram_10samples",	"SefCheckGPT_max_ngram_10_samples",	"SefCheckGPT_nli_10samples"],
                     "transformation": return_original},
    "LMvsLM": {"columns": ["LMvsLM_label"],
               "transformation": transform_factual_to_int},
    "SAC3": {"columns": ["sc2_score", "sac3_q_score", "sac3_qm(falcon)_score", "sac3_qm(starling)_score"], #, "sac3_score(all)"
             "transformation": return_original},
    "AlignScorer": {"columns": ["AlignScore-base", "AlignScore-large"],
                    "transformation": invert_prob},
    "ScaleScorer": {"columns": ["ScaleScorer-large", "ScaleScorer-xl"],
                    "transformation": invert_prob},
}

config_benchmarks = {
"SelfCheckGPT": {
    "reference_binary": False,
    "transformation_first": calculate_average_label,
    "transformation_labels": transform_prob
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

def plot_boxplots(df, col_name):
    # Create a new column to represent the condition for boxplots
    df['condition'] = df['labels'].apply(lambda x: 'Label 0' if x == 0 else 'Label 1')

    # Define the order of datasets for x-axis
    dataset_order = df['dataset_name'].unique()

    # Set seaborn style
    sns.set(style="ticks")

    # Plot boxplots using seaborn
    plt.figure(figsize=(13, 8))
    ax = sns.boxplot(data=df, x='dataset_name', y=col_name, hue='condition', order=dataset_order,
                     palette='pastel')
    #plt.title(f'Distribution of {col_name} scores for each dataset', fontsize=16)
    plt.xlabel('Dataset')
    plt.ylabel(f"{col_name} score")

    plt.legend(title='Labels')
    sns.despine(offset=10, trim=True)
    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(dataset_order, rotation=45, horizontalalignment='right')

    plt.tight_layout()
    plt.savefig(f'outputs/{col_name}_boxplots.png')


for score in ["AlignScorer", "ScaleScorer"]: #
    df_aggregated = pd.DataFrame()
    for dataset in ["SelfCheckGPT", "SelfCheckGPT_alternative", "PHD_wiki_1y", "PHD_wiki_10w", "PHD_wiki_1000w", "FactScore_PerplexityAI", "FactScore_InstructGPT", "FactScore_ChatGPT", "BAMBOO_abshallu_4k", "BAMBOO_abshallu_16k", "BAMBOO_senhallu_4k", "BAMBOO_senhallu_16k", "ScreenEval_longformer", "ScreenEval_gpt4", "ScreenEval_human", "HaluEval_summarization_data", "HaluEval_dialogue_data", "HaluEval_qa_data"]: #  "FAVA_llama", "FAVA_chatgpt", "FELM_math", "FELM_reasoning", "FELM_science", "FELM_wk", "FELM_writing_rec"
        if score == "SelfCheckGPT" and "SelfCheckGPT" in dataset:
            path_to_df = os.path.join("outputs", dataset, f"{score}_updated_data_ngram.csv")
        else:
            path_to_df = os.path.join("outputs", dataset, f"{score}_updated_data.csv")
        print(f"Dataset name: {dataset}.")
        method = score
        benchmark = dataset.split("/")[-1].split("_")[0]
        df = get_data(path_to_df, method, benchmark)
        df['dataset_name'] = dataset
        if "PHD" in dataset:
            df["dataset_name"] = dataset_names_phd[dataset]
        df = df.dropna(subset=["labels"])
        df["labels"] = df["labels"].apply(lambda x: int(x>0.5))

        df_aggregated = pd.concat([df_aggregated, df[config[method]["columns"] + ['dataset_name', 'labels']]], ignore_index=True)

    print(df_aggregated)
    print(df_aggregated['labels'].unique())
    for col_name in config[score]["columns"]:
        plot_boxplots(df_aggregated, col_name)


