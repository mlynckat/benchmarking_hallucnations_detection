import json
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset


class ReadData:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None

    def load_jsonl_data(self):
        """
        Load json data
        """
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        self.data = pd.DataFrame(data)

    def load_hf_data(self, split, subset=None):
        """
        Load json data
        """
        if subset:
            dataset = load_dataset(self.data_path, subset)
        else:
            dataset = load_dataset(self.data_path)

        print(dataset)

        self.data = pd.DataFrame(dataset[split])

    def read_data(self):
        """
        Abstract method to read specific columns from the loaded data.
        """

        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() before read_data().")

        column_mapping = {
            self.references_col: "references",
            self.generations_col: "generations",
            self.labels_col: "labels",
            self.query_col: "query",
            self.correct_answer_col: "correct_answer"
        }

        for old_name, new_name in column_mapping.items():
            if new_name is not None:
                self.data.rename(columns={old_name: new_name}, inplace=True)
        assert len(self.data.columns) == len(set(self.data.columns)), "Duplicate column names detected."
        return self.data



class SelfCheckGPTData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = 'wiki_bio_text'
        self.generations_col = 'gpt3_text'
        self.labels_col = 'annotation'
        self.query_col = 'wiki_bio_test_idx'
        self.correct_answer_col = 'gpt3_text_samples'

    def load(self):
        self.load_hf_data("evaluation")


class HaluEvalData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.data_path = Path(self.data_path)
        self.fields = {
            "qa_data": {
                "reference": "knowledge",
                "query": "question",
                "generations": "hallucinated_answer",
                "labels": None,
                "correct_answer": "right_answer",
            },
            "dialogue_data": {
                "reference": "knowledge",
                "query": "dialogue_history",
                "generations": "hallucinated_response",
                "labels": None,
                "correct_answer": "right_response",
            },
            "summarization_data": {
                "reference": "document",
                "query": None,
                "generations": "hallucinated_summary",
                "labels": None,
                "correct_answer": "right_summary ",
            },
            "general_data": {
                "reference": None,
                "query": "user_query",
                "generations": "chatgpt_response",
                "labels": "hallucination_label",
                "correct_answer": None,
            }

        }
        data_type = self.data_path.stem
        self.references_col = self.fields[data_type]['reference']
        self.generations_col = self.fields[data_type]['generations']
        self.labels_col = self.fields[data_type]['labels']
        self.query_col = self.fields[data_type]['query']
        self.correct_answer_col = self.fields[data_type]['correct_answer']

    def load(self):
        self.load_jsonl_data()


class BAMBOOData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = 'content'
        self.generations_col = 'hypothesis'
        self.labels_col = 'answer'
        self.query_col = None
        self.correct_answer_col = None

    def load(self):
        self.load_jsonl_data()


class FELMData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = None
        self.generations_col = 'segmented_response'
        self.labels_col = 'labels'
        self.query_col = 'prompt'
        self.correct_answer_col = 'comment'

    def load(self, subset):
        self.load_hf_data("test", subset)


class PHDData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = None
        self.generations_col = 'AI'
        self.labels_col = 'label'
        self.query_col = 'entity'
        self.correct_answer_col = 'comment'

    def load(self, subset):

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = pd.DataFrame(data[subset])

class FactScoreData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = None
        self.generations_col = 'output'
        self.labels_col = 'label'
        self.query_col = 'input'
        self.correct_answer_col = 'comment'

    def load(self):
        self.load_jsonl_data()


class ExpertQAData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = None
        self.generations_col = 'answer_string'
        self.labels_col = 'claims'
        self.query_col = 'question'
        self.correct_answer_col = ''

    def load(self, model):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                temp = json.loads(line)
                try:
                    print(temp["answers"][model].keys())
                    temp["answers"][model]["question"] = temp["question"]
                    data.append(temp["answers"][model])
                except KeyError:
                    print(f"KeyError for {model}")

class BUMPData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = 'article'
        self.generations_col = 'edited_summary'
        self.labels_col = 'error_type'
        self.query_col = None
        self.correct_answer_col = 'reference_summary'

    def load(self):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data_temp = json.load(f)
        for line in data_temp:
            data.append(line)

        self.data = pd.DataFrame(data)

        print(self.data.head())
        print(self.data.columns)
        print(self.data[['error_type', 'corrected_error_type']])


class FAVAData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = None
        self.generations_col = 'output'
        self.labels_col = 'annotated'
        self.query_col = 'prompt'
        self.correct_answer_col = None

    def load(self, model):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data_temp = json.load(f)

        for line in data_temp:
            if line["model"] == model:
                data.append(line)

        self.data = pd.DataFrame(data)

        print(self.data.head())
        print(self.data.columns)

class FacToolData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = None
        self.generations_col = 'response'
        self.labels_col = 'label'
        self.query_col = 'prompt'
        self.correct_answer_col = None

    def load(self):
        self.load_jsonl_data()

class QAGSData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = 'article'
        self.generations_col = None
        self.labels_col = None
        self.query_col = None
        self.correct_answer_col = None

    @staticmethod
    def most_frequent_response(responses):
        # Extracting only the 'response' values from the list of dictionaries
        response_values = [item['response'] for item in responses]

        # Counting the occurrences of each response
        response_counts = Counter(response_values)

        # Finding the most common response and its count
        most_common_response, count = response_counts.most_common(1)[0]

        return most_common_response

    def load(self, granularity="sentence"):
        data_articles = []
        data_sentences = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                one_line = json.loads(line)
                article = {"generated_summary": "", "article": one_line["article"], "label": None}
                labels_in_article = []
                print(f"Article: {one_line['article']}")
                for sentence in one_line["summary_sentences"]:
                    print(f"Sentence: {sentence['sentence']}")
                    print(f"Responses: {sentence['responses']}")
                    article["generated_summary"] += sentence["sentence"]+" "
                    sentence["label"] = self.most_frequent_response(sentence["responses"])
                    labels_in_article.append(sentence["label"])
                    sentence["article"] = one_line["article"]
                    data_sentences.append(sentence)
                article["label"] = "no" if "no" in labels_in_article else "yes"
                data_articles.append(article)

                print(len(data_sentences))
                print(len(data_articles))

        if granularity == "sentence":
            self.data = pd.DataFrame(data_sentences)
        elif granularity == "article":
            self.data = pd.DataFrame(data_articles)
        else:
            raise ValueError("Unsupported granularity. Expecting 'sentence' or 'article'.")


        print(self.data["label"].value_counts())
        print(self.data)
        print(self.data.columns)


# Example usage:
if __name__ == "__main__":
    for path_to_dataset in Path("data/QAGS").glob("*.jsonl"):
        data = QAGSData(path_to_dataset)
        data.load("article")
        output = data.read_data()
