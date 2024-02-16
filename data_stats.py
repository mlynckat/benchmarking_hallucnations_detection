import json
import jsonlines
from main import load_data_by_name
from transformers import GPT2Tokenizer

# Load config.json
with open("config.json") as json_file:
    config = json.load(json_file)

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


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

    for dataset_name in ["SelfCheckGPT", "PHD", "FactScore",  "BAMBOO", "ScreenEval", "HaluEval", "FELM", "FAVA"]:
        data = load_data_by_name(dataset_name)

        for current_data_name in data.keys():
            current_data = data[current_data_name]
            current_data_no_na = current_data.dropna(subset=["generations", "labels"])

            try:
                current_data_no_dup = current_data_no_na.drop_duplicates(subset=["query", "generations"])
            except KeyError:
                current_data_no_dup = current_data_no_na.drop_duplicates(subset=["references", "generations"])

            size = current_data_no_dup.shape[0]
            labels = (current_data_no_dup["labels"] > 0.5).astype(int)
            ratio_hallucinated_samples = sum(labels) / len(labels)

            # Tokenize generations and query
            current_data_no_dup["generations_tokens"] = current_data_no_dup["generations"].apply(
                lambda x: len(tokenizer.encode(x)))
            current_data_no_dup["query_tokens"] = current_data_no_dup.apply(lambda row: len(tokenizer.encode(
                config[current_data_name]['prompt'] + " " + row[
                    "references"])) if "BAMBOO" in current_data_name or current_data_name == "HaluEval_summarization_data" or "ScreenEval" in current_data_name else len(
                tokenizer.encode(config[current_data_name]['prompt'] + " " + row["query"])), axis=1)

            # Calculate average length
            average_length_response = current_data_no_dup["generations_tokens"].mean()
            average_length_query = current_data_no_dup["query_tokens"].mean()

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
