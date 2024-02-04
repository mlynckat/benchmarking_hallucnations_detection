import json

import jsonlines

from main import load_data_by_name

# load config.json
with open("config.json") as json_file:
    config = json.load(json_file)

def create_empty_jsonl_file(file_path):
    # Create an empty JSONL file
    with open(file_path, mode='w') as file:
        pass


def add_entry_to_jsonl(file_path, new_entry):
    # Append a new entry to the existing JSONL file
    with jsonlines.open(file_path, mode='a') as writer:
        writer.write(new_entry)

def main():

    jsonl_file_path = 'outputs/data_stats.jsonl'

    create_empty_jsonl_file(jsonl_file_path)

    for dataset_name in ["HaluEval", "SelfCheckGPT", "FELM", "BAMBOO", "PHD", "FAVA", "FactScore"]:
        data = load_data_by_name(dataset_name)

        for current_data_name in data.keys():
            current_data = data[current_data_name]
            print(current_data.shape[0])
            current_data_no_na = current_data.dropna(subset=["generations"])
            print(current_data_no_na.shape[0])
            print(current_data_no_na[~current_data_no_na.index.isin(current_data.index)]["generations"])
            try:
                current_data_no_dup = current_data_no_na.drop_duplicates(subset=["query", "generations"])
            except KeyError:
                current_data_no_dup = current_data_no_na.drop_duplicates(subset=["references", "generations"])
            print(current_data_no_dup.shape[0])
            print(current_data_no_na[~current_data_no_na.index.isin(current_data_no_dup.index)]["generations"])
            size = current_data_no_dup.shape[0]
            labels = (current_data_no_dup["labels"] > 0.5).astype(int)
            # ratio of labels=1
            ratio_hallucinated_samples = sum(labels) / len(labels)
            average_length_response = current_data_no_dup["generations"].apply(lambda x: len(x.split())).mean()
            prompt = config[current_data_name]['prompt']
            if "BAMBOO" in current_data_name or current_data_name == "HaluEval_summarization_data":
                full_query = prompt + " " + current_data_no_dup["references"]
            elif current_data_name == "HaluEval_dialogue_data" or current_data_name == "HaluEval_qa_data":
                full_query = prompt  + " " + current_data["query"] + " " + current_data["references"]
            else:
                full_query = prompt + " " + current_data["query"]
            average_length_query = full_query.apply(lambda x: len(x.split())).mean()

            new_entry = {
                "dataset_name": current_data_name,
                "size": size,
                "%_hallucinated_samples": round(ratio_hallucinated_samples, 2),
                "average_length_query": round(average_length_query, 1),
                "average_length_response": round(average_length_response, 1)
            }

            # Add the new entry to the JSONL file
            add_entry_to_jsonl(jsonl_file_path, new_entry)

if __name__ == "__main__":
    main()
