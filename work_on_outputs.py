import ast
import os
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
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
    "SAC3": {"columns": ["sc2_score_short", "sac3_q_score_short", "sac3_qm(falcon)_score_short", "sac3_qm(starling)_score_short"],
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
    # get number of samples with labels==1

    #print(f"Dropped: {na_free[~na_free.index.isin(duplicates_free.index)]['generations']}")
    return duplicates_free


def calculate_correlation(numerical_labels, predictions):
    """
    Calculate Pearson and Spearman correlation coefficients between two columns of a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - num_labels_col: str, the name of the column with numerical labels
    - predictions: str, the name of the column with predictions

    Returns:
    - pearson_corr: float, Pearson correlation coefficient
    - spearman_corr: float, Spearman correlation coefficient
    """
    # Calculate Pearson correlation coefficient
    pearson_corr, _ = pearsonr(numerical_labels, predictions)

    # Calculate Spearman correlation coefficient
    spearman_corr, _ = spearmanr(numerical_labels, predictions)

    return pearson_corr, spearman_corr


def calculate_binary_metrics(labels, predictions, benchmark, method, col_name):
    """
    Calculate accuracy, precision, recall, and F1 score for binary classification.

    Parameters:
    - labels: pd.Series, array of labels
    - predictions: pd.Series, array of predictions

    """

    reference_binary_int = config_benchmarks[benchmark]["reference_binary"]
    if not reference_binary_int:
        labels = labels.apply(config_benchmarks[benchmark]["transformation_labels"])

    #print number of labels==1
    print(f"Number of samples with labels==1: {labels[labels==1].shape[0]}")
    predictions = predictions.apply(config[method]["transformation"])
    #compare to threshold and transform to binary
    predictions = predictions.apply(lambda x: int(x > config[method]["thresholds"][col_name]))

    # Calculate evaluation metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1_score_val = f1_score(labels, predictions)
    # print classification report
    #print(f"Classification report for {method}:")
    #print(classification_report(labels, predictions))

    return accuracy, precision, recall, f1_score_val


def calculate_aucroc(labels, predictions, benchmark, method):
    """
    Calculate Pearson and Spearman correlation coefficients between two columns of a DataFrame,
    and compute accuracy, precision, recall, and F1 score after transforming values in the first column.

    Parameters:
    - labels: pd.Series, array of labels
    - predictions: pd.Series, array of predictions
    - reference_binary_int: bool, whether the reference labels are binary integers
    - prediction_inverted: bool, whether the predictions are inverted

    Returns:
    - ROC-AUC score: float, ROC-AUC score after transformation
    """

    reference_binary_int = config_benchmarks[benchmark]["reference_binary"]
    if not reference_binary_int:
        labels = labels.apply(config_benchmarks[benchmark]["transformation_labels"])
    #print number of labels==1
    print(f"Number of samples with labels==1: {labels[labels==1].shape[0]}")

    predictions = predictions.apply(config[method]["transformation"])
    try:
        roc_auc_score_val = roc_auc_score(labels, predictions)
    except ValueError:
        roc_auc_score_val = None

    return roc_auc_score_val

def plot_boxplots_for_non_binary(labels, predictions, col_name, benchmark_name, predictions2=None, col_name2=None):
    """
    Plot boxplots for binary labels and non-binary predictions.

    Parameters:
    - labels: pd.Series, array of labels
    - predictions: pd.Series, array of predictions
    """

    labels = (labels > 0.5).astype(int)
    predictions_label_0 = predictions[labels == 0]
    predictions_label_1 = predictions[labels == 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Position for the first set of boxplots
    positions = [1, 2]

    # Create boxplots for the first set of predictions (predictions)
    bp1 = ax.boxplot([predictions_label_0, predictions_label_1], labels=['Factual', 'Non-factual'], positions=positions, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('skyblue')  # Set color for the first set of boxplots

    if predictions2 is not None:
        predictions_label_0_2 = predictions2[labels == 0]
        predictions_label_1_2 = predictions2[labels == 1]

        # Position for the second set of boxplots
        positions2 = [1.3, 2.3]

        # Create boxplots for the second set of predictions (predictions2)
        bp2 = ax.boxplot([predictions_label_0_2, predictions_label_1_2], positions=positions2, widths=0.2, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgreen')  # Set color for the second set of boxplots

    # Set axis labels and title
    ax.set_xlabel('Labels')
    ax.set_ylabel('Predictions')
    ax.set_title('Distribution of scores: ' + col_name + ". " + benchmark_name)

    # Add legend
    if predictions2 is not None:
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Predictions', 'Predictions2'], loc='upper right')

    # Show the plot
    plt.show()


def plot_distribution(df, column1, column2):
    plt.figure(figsize=(10, 6))

    # Plot histogram for column 1
    plt.hist(df[column1], bins=30, alpha=0.5, label=column1)

    # Plot histogram for column 2
    plt.hist(df[column2], bins=30, alpha=0.5, label=column2)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of {} and {}'.format(column1, column2))
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter(df, ref_col, column1, column2):
    plt.figure(figsize=(10, 6))

    # Plot histogram for column 1
    plt.scatter(df[column1], df[ref_col], alpha=0.5, label=column1, color='blue')

    # Plot histogram for column 2
    plt.scatter(df[column2], df[ref_col], alpha=0.5, label=column2, color='red')

    plt.xlabel('Max_logprob')
    plt.ylabel('Label')
    plt.title('Distribution of {} and {}'.format(column1, column2))
    plt.legend()
    plt.grid(True)
    plt.show()

for score in ["SAC3"]:
    for dataset in ["FAVA_llama", "FAVA_chatgpt", "FELM_math", "FELM_reasoning", "FELM_science", "FELM_wk", "FELM_writing_rec"]: #SelfCheckGPT", "SelfCheckGPT_alternative", "PHD_wiki_1y", "PHD_wiki_10w", "PHD_wiki_1000w", "FactScore_PerplexityAI", "FactScore_InstructGPT", "FactScore_ChatGPT", , "FAVA_llama",   "BAMBOO_abshallu_4k", "BAMBOO_abshallu_16k", "BAMBOO_senhallu_4k", "BAMBOO_senhallu_4k", "ScreenEval_longformer", "ScreenEval_gpt4", "ScreenEval_human", "HaluEval_summarization_data", "HaluEval_dialogue_data", "HaluEval_qa_data" "FAVA_chatgpt", "FAVA_llama",  "FELM_math", "FELM_reasoning", "FELM_science", "FELM_wk", "FELM_writing_rec"
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


            # correct_passages = df.loc[df["labels"] == 0, "query"].tolist()
            # print(correct_passages)
            # print(len(correct_passages))
            # print(f"Quantiles: {df['labels'].quantile([0.25, 0.5, 0.75])}")
            #
            # # parse as list of lists
            # df['correct_answer'] = df['correct_answer'].apply(ast.literal_eval)
            # df['additional_samples_gpt3'] = df['additional_samples_gpt3'].apply(ast.literal_eval)
            #
            # len_original = df['correct_answer'].apply(lambda x: sum([len(sample.split()) for sample in x])/len(x)).mean()
            # len_new = df['additional_samples_gpt3'].apply(lambda x: sum([len(sample.split()) for sample in x])/len(x)).mean()
            #
            # print(f"Lenght of original vs new: {len_original} vs {len_new}")
            #
            # num_original = df['correct_answer'].apply(lambda x: len(x)).mean()
            # num_new = df['additional_samples_gpt3'].apply(lambda x: len(x)).mean()
            #
            # print(f"Num of original vs new: {num_original} vs {num_new}")
            #
            # plot_distribution(df, "SefCheckGPT_max_ngram", "SefCheckGPT_max_ngram_original")
            # plot_scatter(df, "labels", "SefCheckGPT_max_ngram", "SefCheckGPT_max_ngram_original")
            # plot_boxplots_for_non_binary(df["labels"], df["SefCheckGPT_max_ngram"], "SefCheckGPT_max_ngram", path.split("/")[-1], predictions2=df["SefCheckGPT_max_ngram_original"], col_name2="SefCheckGPT_max_ngram")

        for pred_col in config[method]["columns"]:
            if "labels" not in df.columns or pred_col not in df.columns:
                raise ValueError("Specified columns not found in the DataFrame.")

            else:

                #pearson, spearman = calculate_correlation(df["labels"], df[pred_col])
                #print(f"Pearson correlation between labels and {pred_col}: {pearson}")
                #print(f"Spearmann correlation between labels and {pred_col}: {spearman}")
                rocauc = calculate_aucroc(df["labels"], df[pred_col], benchmark, method)
                try:
                    print(f"ROC-AUC score between labels and {pred_col}: {round(rocauc, 2)}")
                except:
                    print(f"ROC-AUC score between labels and {pred_col}: {rocauc}")

                #accuracy, precision, recall, f1_score_val = calculate_binary_metrics(df["labels"], df[pred_col], benchmark, method, pred_col)
                #print(f"Accuracy between labels and {pred_col}: {round(accuracy, 2)}")
                #print(f"Precision between labels and {pred_col}: {round(precision, 2)}")
                #print(f"Recall between labels and {pred_col}: {round(recall, 2)}")
                #print(f"F1 score between labels and {pred_col}: {round(f1_score_val, 2)}")
                #plot_boxplots_for_non_binary(df["labels"], df[pred_col], pred_col, path.split("/")[-1])
                print("-"*50)

        # if benchmark == "FactScore":
        #     df["categories"] = df["cat"].apply(lambda x: x.split(",")[0].replace("[", "").replace("'", ""))
        #     categories = df["categories"].unique()
        #
        #
        #     for category in categories:
        #         df_subset = df[df["categories"] == category]
        #         print(f"Dataset name: {dataset} {category}")
        #         print(f"The size of the dataset is {df_subset.shape[0]}")
        #
        #         for pred_col in config[method]["columns"]:
        #             if "labels" not in df_subset.columns or pred_col not in df_subset.columns:
        #                 raise ValueError("Specified columns not found in the DataFrame.")
        #             else:
        #
        #                 # pearson, spearman = calculate_correlation(df["labels"], df[pred_col])
        #                 # print(f"Pearson correlation between labels and {pred_col}: {pearson}")
        #                 # print(f"Spearmann correlation between labels and {pred_col}: {spearman}")
        #                 rocauc = calculate_aucroc(df_subset["labels"], df_subset[pred_col], benchmark, method)
        #                 try:
        #                     print(f"ROC-AUC score between labels and {pred_col}: {round(rocauc, 2)}")
        #                 except:
        #                     print(f"ROC-AUC score between labels and {pred_col}: {rocauc}")
        #                 accuracy, precision, recall, f1_score_val = calculate_binary_metrics(df_subset["labels"], df_subset[pred_col],
        #                                                                                      benchmark, method, pred_col)
        #                 print(f"Accuracy between labels and {pred_col}: {round(accuracy, 2)}")
        #                 print(f"Precision between labels and {pred_col}: {round(precision, 2)}")
        #                 print(f"Recall between labels and {pred_col}: {round(recall, 2)}")
        #                 #print(f"F1 score between labels and {pred_col}: {round(f1_score_val, 2)}")
        #                 print("-" * 50)
        #

