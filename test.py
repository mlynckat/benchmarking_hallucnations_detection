import json
import os
import logging

from main import load_data_by_name
from methods_implementations import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# load config.json
with open("config.json") as json_file:
    config = json.load(json_file)


def get_query(row, benchmark):
    if benchmark == "SelfCheckGPT":
        query = config[benchmark]['prompt'].format(query=row['query'])
    elif "PHD" in benchmark:
        query = config[benchmark]['prompt'].format(entity=row['query'])
    elif "FactScore" in benchmark:
        query = config[benchmark]['prompt'].format(prompt=row['query'])
    elif "BAMBOO" in benchmark:
        query = config[benchmark]['prompt'].format(passage=row['references'])
    elif "HaluEval" in benchmark:
        if "dialogue_data" in benchmark:
            query = config[benchmark]['prompt'].format(message_history=row['query'], knowledge=row['references'])
        elif "qa_data" in benchmark:
            query = config[benchmark]['prompt'].format(question=row['query'], passage=row['references'])
        elif "summarization" in benchmark:
            query = config[benchmark]['prompt'].format(passage=row['references'])
        elif "general_data" in benchmark:
            query = config[benchmark]['prompt'].format(user_query=row['query'])
        else:
            raise ValueError("Unknown HaluEval subset")
    elif "FELM" in benchmark:
        query = config[benchmark]['prompt'].format(prompt=row['query'])
    elif "FAVA" in benchmark:
        query = config[benchmark]['prompt'].format(prompt=row['query'])
    else:
        raise ValueError("Unknown benchmark")
    return query


def apply_method(row, method, benchmark):
    task = config[benchmark]['task']
    if method.__class__.__name__ == "SelfCheckGPT":
        query = get_query(row, benchmark)
        model_name = config[benchmark]['model_name']
        predictions = method.make_predictions(row, query=query, model_name=model_name)
    elif method.__class__.__name__ == "LMvsLM" or method.__class__.__name__ == "SAC3":
        query = get_query(row, benchmark)
        predictions = method.make_predictions(row, query=query, task=task)
    elif method.__class__.__name__ == "AlignScorer":
        query = None
        predictions = method.make_predictions(row, query=query)
    else:
        raise ValueError("Unknown method")
    return predictions


for dataset_name in ["felm", "halueval", "bamboo", "phd", "felm", "fava", "factscore"]:  # , "selfcheckgpt", "bamboo", "phd", "felm", "fava"
    data = load_data_by_name(dataset_name)

    for current_data_name in data.keys():
        logging.info(f"Working on {current_data_name}...")

        output_dir = f"outputs/{current_data_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        current_data = data[current_data_name]

        for method in [SAC3(), LMvsLM()]: #, SAC3(), LMvsLM() , SelfCheckGPT(), LMvsLM(), AlignScorer()
            logging.info(f"Working on {method.__class__.__name__}...")

            if method.__class__.__name__ == "SAC3":
                if config[current_data_name]["task"] == "summarization" or config[current_data_name][
                    "task"] == "dialogue":
                    continue
            elif method.__class__.__name__ == "AlignScorer":
                if config[current_data_name]["task"] == "general_qa":
                    continue
                else:
                    assert "references" in current_data.columns, "References column not found in data"

            elif method.__class__.__name__ == "LMvsLM":
                if config[current_data_name]["task"] == "summarization" or config[current_data_name][
                    "task"] == "dialogue":
                    continue



            updated_data = current_data.loc[0:5].apply(lambda row: apply_method(row, method, current_data_name),
                                                       axis=1).reset_index(drop=True)
            updated_data.to_csv(os.path.join(output_dir, f"{method.__class__.__name__}_updated_data.csv"),
                                encoding="utf-8", index=False)
