import json
import os
import logging
import argparse
import sys

import pandas as pd

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

methods_map = {
    "sac3": SAC3,
    "lmvslm": LMvsLM,
    "alignscorer": AlignScorer,
    "selfcheckgpt": SelfCheckGPT,
    "scalescorer": ScaleScorer
}

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
    elif "ScreenEval" in benchmark:
        query = config[benchmark]['prompt'].format(prompt=row['query'], passage=row['references'])
    else:
        raise ValueError("Unknown benchmark")
    return query


def apply_method(row, method, benchmark):
    task = config[benchmark]['task']
    if method.__class__.__name__ == "SelfCheckGPT" or method.__class__.__name__ == "SAC3":
        query = get_query(row, benchmark)
        model_name = config[benchmark]['model_name']
        predictions = method.make_predictions(row, query=query, model_name=model_name)
    elif method.__class__.__name__ == "LMvsLM":
        query = get_query(row, benchmark)
        predictions = method.make_predictions(row, query=query, task=task)
    elif method.__class__.__name__ == "AlignScorer" or method.__class__.__name__ == "ScaleScorer":
        query = None
        predictions = method.make_predictions(row, query=query)
    else:
        raise ValueError("Unknown method")
    return pd.DataFrame(predictions).transpose()

def main(benchmarks, methods):  # Added main function for better organization
    for dataset_name in benchmarks: #["selfcheckgpt", "felm", "bamboo", "phd", "felm", "fava", "factscore"]
        data = load_data_by_name(dataset_name)

        for current_data_name in data.keys():
            logging.info(f"Working on {current_data_name}...")

            output_dir = f"outputs/{current_data_name}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            current_data = data[current_data_name]

            for method in methods:
                method = methods_map[method.lower()]()
                logging.info(f"Working on {method.__class__.__name__}...")

                if method.__class__.__name__ == "SAC3":
                    if config[current_data_name]["task"] == "summarization" or config[current_data_name][
                        "task"] == "dialogue":
                        continue
                elif method.__class__.__name__ == "AlignScorer" or method.__class__.__name__ == "ScaleScorer":
                    if config[current_data_name]["task"] == "general_qa":
                        continue
                    else:
                        assert "references" in current_data.columns, "References column not found in data"

                elif method.__class__.__name__ == "LMvsLM":
                    if config[current_data_name]["task"] == "summarization" or config[current_data_name][
                        "task"] == "dialogue":
                        continue

                # check if file exists
                if os.path.exists(os.path.join(output_dir, f"{method.__class__.__name__}_updated_data_add.csv")):
                    updated_data = pd.read_csv(
                        os.path.join(output_dir, f"{method.__class__.__name__}_updated_data_add.csv"),
                        encoding="utf-8")
                    updated_data_exists = True
                else:
                    updated_data_exists = False

                for ind, row in current_data.iterrows():
                    if updated_data_exists and row['generations'] in updated_data['generations'].values:
                        logging.info(f"Skipping {row['generations']}")
                        continue
                    else:
                        logging.info(f"Working on {row['generations']}")
                        row_data = apply_method(row, method, current_data_name)
                        if updated_data_exists:
                            updated_data = pd.concat([updated_data, row_data], ignore_index=True)
                        else:
                            updated_data = row_data
                            updated_data_exists = True
                        updated_data.to_csv(os.path.join(output_dir, f"{method.__class__.__name__}_updated_data_add.csv"),
                                            encoding="utf-8", index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the benchmarking script with specified benchmarks and methods.")
    parser.add_argument("--benchmarks", nargs="+", type=str,
                        help="Specify the benchmarks (e.g., SelfCheckGPT, FELM, etc.)")
    parser.add_argument("--methods", nargs="+", type=str, help="Specify the methods (e.g., SAC3, LMvsLM, etc.)")

    args = parser.parse_args()

    if not args.benchmarks or not args.methods:
        print("Both --benchmarks and --methods are required.")
    else:
        main(args.benchmarks, args.methods)

    # Example:
    # python3 test.py --benchmarks SelfCheckGPT FELM --methods SAC3 LMvsLM

