import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# read pandas dataframe from csv

path_to_df = "outputs/SelfCheckGPT/SAC3_updated_data.csv"

df = pd.read_csv(path_to_df)
print(f"The size of the dataset is {df.shape[0]}")
# drop duplicates
df = df.drop_duplicates(subset=['query'], keep='first')
print(f"The size of the dataset is {df.shape[0]}")
# drop rows with nan values
df = df.dropna()

print(f"The size of the dataset is {df.shape[0]}")

def calculate_correlation(df, col1, col2):
    """
    Calculate Pearson and Spearman correlation coefficients between two columns of a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - col1: str, the name of the first column
    - col2: str, the name of the second column

    Returns:
    - pearson_corr: float, Pearson correlation coefficient
    - spearman_corr: float, Spearman correlation coefficient
    """
    # Check if the specified columns exist in the DataFrame
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Specified columns not found in the DataFrame.")

    # Extract the specified columns
    data_col1 = df[col1]
    data_col2 = df[col2]

    # Calculate Pearson correlation coefficient
    pearson_corr, _ = pearsonr(data_col1, data_col2)

    # Calculate Spearman correlation coefficient
    spearman_corr, _ = spearmanr(data_col1, data_col2)

    return pearson_corr, spearman_corr
for col in ["sc2_score", "sac3_q_score", "sac3_qm(falcon)_score", "sac3_qm(starling)_score"]:
    print(col)
    pearson, spearman = calculate_correlation(df, "labels", col)
    print(f"For {col}, Pearson correlation coefficient is {pearson} and Spearman correlation coefficient is {spearman}")



def calculate_other_metrics(df, col1, col2, transform_reference=False, hypothesis_factual=False):
    """
    Calculate Pearson and Spearman correlation coefficients between two columns of a DataFrame,
    and compute accuracy, precision, recall, and F1 score after transforming values in the first column.

    Parameters:
    - df: pandas DataFrame
    - col1: str, the name of the first column
    - col2: str, the name of the second column

    Returns:
    - accuracy: float, accuracy score after transformation
    - precision: float, precision score after transformation
    - recall: float, recall score after transformation
    - f1_score_val: float, F1 score after transformation
    """
    # Check if the specified columns exist in the DataFrame
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Specified columns not found in the DataFrame.")

    # Extract the specified columns
    data_col1 = df[col1]
    data_col2 = df[col2]

    # if not 0 or 1 in transformed_col1 row, remove it
    #data_col2 = data_col2[data_col1.isin([0, 1])]
    #data_col1 = data_col1[data_col1.isin([0, 1])]

    # Transform values in the first column to 1 or 0
    if transform_reference:
        transformed_col1 = (data_col1 > 0.5).astype(int)
    else:
        transformed_col1 = data_col1.astype(int)

    factual_dict = {"factual": 0, "non-factual": 1}

    if hypothesis_factual:
        transformed_col2 = data_col2.map(factual_dict)
    else:
        transformed_col2 = data_col2

    print(transformed_col1)
    print(transformed_col2)


    # Calculate Pearson correlation coefficient
    pearson_corr, _ = pearsonr(transformed_col1, transformed_col2)

    # Calculate Spearman correlation coefficient
    spearman_corr, _ = spearmanr(transformed_col1, transformed_col2)

    # Calculate evaluation metrics
    accuracy = accuracy_score(transformed_col1, transformed_col2)
    precision = precision_score(transformed_col1, transformed_col2)
    recall = recall_score(transformed_col1, transformed_col2)
    f1_score_val = f1_score(transformed_col1, transformed_col2)

    return pearson_corr, spearman_corr, accuracy, precision, recall, f1_score_val

# Example usage:
# Assuming you have a DataFrame named 'my_dataframe' with columns 'column1' and 'column2'
# Replace 'my_dataframe', 'column1', and 'column2' with your actual DataFrame and column names

"""for col in ["SefCheckGPT_mqag", "SefCheckGPT_bertscore", "SefCheckGPT_ngram", "SefCheckGPT_nli", "SefCheckGPT_prompting"]:
    print(col)
    pearson_corr, spearman_corr, accuracy, precision, recall, f1_score_val = calculate_other_metrics(df, 'labels', col)
    print(f"Pearson correlation coefficient: {pearson_corr}")
    print(f"Spearman correlation coefficient: {spearman_corr}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score_val}")"""


"""pearson_corr, spearman_corr, accuracy, precision, recall, f1_score_val = calculate_other_metrics(df.loc[6:], 'labels', "LMvsLM_label", transform_reference=False, hypothesis_factual=True)
print(f"Pearson correlation coefficient: {pearson_corr}")
print(f"Spearman correlation coefficient: {spearman_corr}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score_val}")
"""

    #pearson, spearman = calculate_correlation(df, "labels", col)
    #print(f"For {col}, Pearson correlation coefficient is {pearson} and Spearman correlation coefficient is {spearman}")

