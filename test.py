import pandas as pd

from read_load_data import *
from main import load_data_by_name
from methods_implementations import *
from datasets import load_dataset

dataset = load_dataset("wiki_bio")

wiki_bio = dataset["test"]
wiki_bio = pd.DataFrame(wiki_bio)

# load config.json
with open("config.json") as json_file:
    config = json.load(json_file)


# Define the SelfCheckGPT method
def apply_selfcheckgpt(row, method, benchmark, task="hallucination"):
    if benchmark == "selfcheckgpt":
        query = config[benchmark]['prompt'].format(wiki_bio.loc[row['query']]['input_text']['context'])

    elif "PHD" in benchmark:
        query = config[benchmark]['prompt'].format(row['query'])
    elif "BAMBOO" in benchmark:
        query = config[benchmark]['prompt'].format(row['reference'])
    elif "HaluEval" in benchmark:
        if "dialogue_data" in benchmark:
            query = config[benchmark]['prompt'].format(message_history=row['query'], knowledge=row['reference'])
        elif "qa_data" in benchmark:
            query = config[benchmark]['prompt'].format(question=row['query'], passage=row['reference'])
        elif "summarization" in benchmark:
            query = config[benchmark]['prompt'].format(row['reference'])
        elif "general_data" in benchmark:
            query = config[benchmark]['prompt'].format(row['query'])
        else:
            raise ValueError("Unknown HaluEval subset")
    elif "FELM" in benchmark:
        query = config[benchmark]['prompt'].format(row['query'])
    else:
        raise ValueError("Unknown benchmark")
    model_name = config[benchmark]['model_name']
    predictions = method.make_predictions(row, query, model_name)

    return predictions


def apply_lm_vs_lm(row, method, benchmark):
    if benchmark == "SelfCheckGPT":
        query = config[benchmark]['prompt'].format(row['query'])
    elif "PHD" in benchmark:
        query = config[benchmark]['prompt'].format(row['query'])
    elif "BAMBOO" in benchmark:
        query = config[benchmark]['prompt'].format(row['reference'])
    elif "HaluEval" in benchmark:
        if "dialogue_data" in benchmark:
            query = config[benchmark]['prompt'].format(message_history=row['query'], knowledge=row['reference'])
        elif "qa_data" in benchmark:
            query = config[benchmark]['prompt'].format(question=row['query'], passage=row['reference'])
        elif "summarization" in benchmark:
            query = config[benchmark]['prompt'].format(row['reference'])
        elif "general_data" in benchmark:
            query = config[benchmark]['prompt'].format(row['query'])
        else:
            raise ValueError("Unknown HaluEval subset")
    elif "FELM" in benchmark:
        query = config[benchmark]['prompt'].format(row['query'])
    else:
        raise ValueError("Unknown benchmark")

    predictions = method.make_predictions(row, query)
    return predictions

for dataset_name in ["selfcheckgpt", "phd", "felm"]:
    data = load_data_by_name(dataset_name)

    for current_data_name in data:

        output_dir = f"outputs/{current_data_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        current_data = data[current_data_name]

        # Create an empty dataframe to store the updated data
        #updated_data = pd.DataFrame()
        method = SelfCheckGPT()
        # Apply the function to each row in the 'current_data' dataframe
        updated_data = current_data.loc[0:5].apply(lambda row: apply_selfcheckgpt(row, method, dataset_name), axis=1).reset_index(drop=True)
        updated_data.to_csv(os.path.join(output_dir, "SelfCheckGPT_updated_data.csv"), encoding="utf-8", index=False)

        lmvslm = LMvsLM()
        updated_data = current_data.loc[0:5].apply(lambda row: apply_lm_vs_lm(row, lmvslm, dataset_name), axis=1).reset_index(drop=True)
        updated_data.to_csv(os.path.join(output_dir, "LMvsLM_updated_data.csv"), encoding="utf-8", index=False)
