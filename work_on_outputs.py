import os
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Set seaborn style
sns.set_style("white")

def transform_prob(prob_col):
    return (prob_col > 0.5).astype(int)

def invert_and_transform_prob(prob_col):
    return (prob_col < 0.5).astype(int)

config = {
    "SelfCheckGPT": {"columns": ["SefCheckGPT_mqag", "SefCheckGPT_bertscore", "SefCheckGPT_max_ngram", "SefCheckGPT_nli", "SefCheckGPT_prompting"],
                     "transformation": ""},
    "SelfCheckGPT_10samples": {"columns": ["SefCheckGPT_mqag_10samples", "SefCheckGPT_bertscore_10samples",	"SefCheckGPT_ngram_10samples",	"SefCheckGPT_max_ngram_10_samples",	"SefCheckGPT_nli_10samples"],
                     "transformation": ""},
    "LMvsLM": {"columns": ["LMvsLM_label"],
               "transformation": ""},
    "SAC3": {"columns": ["sc2_score", "sac3_q_score", "sac3_qm(falcon)_score", "sac3_qm(starling)_score"],
             "transformation": transform_prob},
    "AlignScorer": {"columns": ["AlignScore-base", "AlignScore-large"],
                    "transformation": invert_and_transform_prob},
}



def get_data(path_to_df, method):
    df = pd.read_csv(path_to_df)
    print(f"The size of the dataset is {df.shape[0]}")
    # drop duplicates
    na_free = df.dropna(subset=config[method]["columns"])
    print(f"The size of the dataset after dropping NA's is {na_free.shape[0]}")
    print(f"Dropped: {df[~df.index.isin(na_free.index)]['generations']}")
    duplicates_free = na_free.drop_duplicates(subset=['generations'], keep='first')
    print(f"The size of the dataset is after removing duplicates: {duplicates_free.shape[0]}")
    print(f"Dropped: {na_free[~na_free.index.isin(duplicates_free.index)]['generations']}")
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


def calculate_binary_metrics(labels, predictions, method, reference_binary_int=False):
    """
    Calculate Pearson and Spearman correlation coefficients between two columns of a DataFrame,
    and compute accuracy, precision, recall, and F1 score after transforming values in the first column.

    Parameters:
    - df: pandas DataFrame
    - labels: pd.Series, array of labels
    - predictions: pd.Series, array of predictions
    - method: str, the name of the method

    Returns:
    - accuracy: float, accuracy score after transformation
    - precision: float, precision score after transformation
    - recall: float, recall score after transformation
    - f1_score_val: float, F1 score after transformation
    - ROC-AUC score: float, ROC-AUC score after transformation
    """

    if not reference_binary_int:
        labels = (labels > 0.5).astype(int)

    predictions = config[method]["transformation"](predictions)

    # Calculate evaluation metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1_score_val = f1_score(labels, predictions)
    roc_auc_score_val = roc_auc_score(labels, predictions)
    # print classification report
    print(f"Classification report for {method}:")
    print(classification_report(labels, predictions))


    return accuracy, precision, recall, f1_score_val, roc_auc_score_val

def plot_boxplots_for_non_binary(labels, predictions, col_name, benchmark_name):
    """
    Plot boxplots for binary labels and non-binary  predictions.

    Parameters:
    - labels: pd.Series, array of labels
    - predictions: pd.Series, array of predictions
    """


    labels = (labels > 0.5).astype(int)
    predictions_label_0 = predictions[labels == 0]
    predictions_label_1 = predictions[labels == 1]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create boxplots for each label
    ax.boxplot([predictions_label_0, predictions_label_1], labels=['Factual', 'Non-factual'])

    # Set axis labels and title
    ax.set_xlabel('Labels')
    ax.set_ylabel('Predictions')
    ax.set_title('Distribution of scores: ' + col_name + ". " + benchmark_name )

    # Show the plot
    plt.show()



for path in ["outputs/SelfCheckGPT"]: #["outputs/BAMBOO_abshallu_4k", "outputs/BAMBOO_abshallu_16k", "outputs/BAMBOO_senhallu_4k", "outputs/BAMBOO_senhallu_4k"]:
    path_to_df = os.path.join(Path(path), "SelfCheckGPT_updated_data_reduced.csv")
    print(path_to_df)
    method = "SelfCheckGPT_10samples"
    df = get_data(path_to_df, method)

    for pred_col in config[method]["columns"]:
        if "labels" not in df.columns or pred_col not in df.columns:
            raise ValueError("Specified columns not found in the DataFrame.")



        else:

            pearson, spearman = calculate_correlation(df["labels"], df[pred_col])
            print(f"Pearson correlation between labels and {pred_col}: {pearson}")
            print(f"Spearmann correlation between labels and {pred_col}: {spearman}")


    """for pred_col in config[method]["columns"]:
        if "labels" not in df.columns or pred_col not in df.columns:
            raise ValueError("Specified columns not found in the DataFrame.")
        else:
            accuracy, precision, recall, f1_score_val, roc_auc_score_value = calculate_binary_metrics(df["labels"], df[pred_col], method)
            print(f"Accuracy between labels and {pred_col}: {accuracy}")
            print(f"Precision between labels and {pred_col}: {precision}")
            print(f"Recall between labels and {pred_col}: {recall}")
            print(f"F1 score between labels and {pred_col}: {f1_score_val}")
            print(f"ROC-AUC score between labels and {pred_col}: {roc_auc_score_value}")
            plot_boxplots_for_non_binary(df["labels"], df[pred_col], pred_col, path.split("/")[-1])
            print("-"*50)"""

