import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
import wikipedia
from bs4 import BeautifulSoup
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)


class ReadData:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None

    def retrieve_wiki_article(self, url):

        r = requests.get(url)
        # Get body content
        soup = BeautifulSoup(r.text, 'html.parser')
        content = soup.find('div', {'id': 'mw-content-text'})
        paragraphs = content.find_all(['p', 'h2', 'li'])
        # Initialize variable
        output = []

        # Iterate through all elements
        for element in paragraphs:
            if "References" in element.text or "Notes" in element.text or "External links" in element.text or "See also" in element.text:
                break
            else:
                output.append(element.text.replace("[edit]", "\n"))
        joint_output = " \n".join(output)
        joint_output = re.sub(r'\s+', ' ', joint_output)
        return joint_output

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

        #for column_name in self.data.columns:
        #    if column_name not in column_mapping.keys():
        #        print(f"None -> {column_name}: {type(self.data.loc[0, column_name])}")

        for old_name, new_name in column_mapping.items():
            if new_name is not None:
                #if old_name is not None:
                #    print(f"{new_name} -> {old_name}: {type(self.data.loc[0, old_name])}")
                #else:
                #    print(f"{new_name} -> {old_name}:{None}")
                self.data.rename(columns={old_name: new_name}, inplace=True)


        assert len(self.data.columns) == len(set(self.data.columns)), f"Duplicate column names detected. {self.data.columns}"

        print(self.data["labels"].head())

        return self.data



class SelfCheckGPTData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = 'wiki_bio_text'
        self.generations_col = 'gpt3_text'
        self.labels_col = 'label'
        self.query_col = 'wiki_bio_test_entity'
        self.correct_answer_col = 'gpt3_text_samples'

        dataset = load_dataset("wiki_bio")
        wiki_bio = dataset["test"]
        self.wiki_bio = pd.DataFrame(wiki_bio)

    def get_entity_name(self, idx):
        entity_name = self.wiki_bio.loc[idx]['input_text']['context']
        return entity_name

    def calculate_average_label(self, input_list):
        # Step 1: Substitute labels
        label_mapping = {"accurate": 0, "minor_inaccurate": 0.5, "major_inaccurate": 1}
        substituted_list = [label_mapping[label] for label in input_list]

        # Step 2: Calculate average
        average_value = sum(substituted_list) / len(substituted_list)

        return average_value

    def load(self):
        self.load_hf_data("evaluation")
        self.data["wiki_bio_test_entity"] = self.data["wiki_bio_test_idx"].apply(lambda x: self.get_entity_name(x))
        self.data["label"] = self.data["annotation"].apply(lambda x: self.calculate_average_label(x))
        # mean amount of words in column "gpt3_text"
        print(self.data["gpt3_text"].str.split().str.len().mean())

class HaluEvalData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.data_path = Path(self.data_path)
        self.fields = {
            "qa_data": {
                "reference": "knowledge",
                "query": "question",
                "generations": "hallucinated_answer",
                "labels": "labels",
                "correct_answer": "right_answer",
            },
            "dialogue_data": {
                "reference": "knowledge",
                "query": "dialogue_history",
                "generations": "hallucinated_response",
                "labels": "labels",
                "correct_answer": "right_response",
            },
            "summarization_data": {
                "reference": "document",
                "query": None,
                "generations": "hallucinated_summary",
                "labels": "labels",
                "correct_answer": "right_summary",
            },
            "general_data": {
                "reference": None,
                "query": "user_query",
                "generations": "chatgpt_response",
                "labels": "labels",
                "correct_answer": None,
            }

        }
        self.data_type = self.data_path.stem
        self.references_col = self.fields[self.data_type]['reference']
        self.generations_col = self.fields[self.data_type]['generations']
        self.labels_col = self.fields[self.data_type]['labels']
        self.query_col = self.fields[self.data_type]['query']
        self.correct_answer_col = self.fields[self.data_type]['correct_answer']

    def transform_labels(self, label):
        if label == "yes":
            return 1
        elif label == "no":
            return 0
        else:
            return None

    def load(self):
        self.load_jsonl_data()
        if self.data_type in ["qa_data", "dialogue_data", "summarization_data"]:
            self.data["labels"] = 1
        elif self.data_type == "general_data":
            self.data["labels"] = self.data["hallucination"].apply(lambda x: self.transform_labels(x))



class BAMBOOData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = 'content'
        self.generations_col = 'hypothesis'
        self.labels_col = 'label'
        self.query_col = None
        self.correct_answer_col = None

    def transform_labels(self, label):
        if label == False:
            return 1
        elif label == True:
            return 0
        else:
            return None

    def load(self):
        self.load_jsonl_data()
        self.data["label"] = self.data["answer"].apply(lambda x: self.transform_labels(x))


class FELMData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = None
        self.generations_col = 'response'
        self.labels_col = 'labels_new'
        self.query_col = 'prompt'
        self.correct_answer_col = 'comment'

    def transform_labels(self, label):
        if label:
            if False in label:
                return 1
            else:
                return 0
        else:
            return None

    def load(self, subset):
        self.load_hf_data("test", subset)
        self.data["labels_new"] = self.data["labels"].apply(lambda x: self.transform_labels(x))
        # rename column labels to labels_old
        self.data.rename(columns={"labels": "labels_old"}, inplace=True)


class PHDData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = 'entity_content'
        self.generations_col = 'AI'
        self.labels_col = 'label_new'
        self.query_col = 'entity'
        self.correct_answer_col = None

    def transform_labels(self, label):
        if label == "factual":
            return 0
        elif label == "non-factual":
            return 1
        else:
            return None
    def clean_text(self, text):
        # Remove multiple spaces and newlines
        cleaned_text = re.sub(r'\s+', ' ', text)

        return cleaned_text

    def load(self, subset):

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = pd.DataFrame(data[subset])
        self.data["entity_content"] = self.data["url"].apply(lambda x: self.retrieve_wiki_article(x))
        #for ind in range(5):
        #   print(self.data.loc[ind, "entity_content"])
        self.data["label_new"] = self.data["label"].apply(lambda x: self.transform_labels(x))


class FactScoreData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = 'article'
        self.generations_col = 'output'
        self.labels_col = 'label'
        self.query_col = 'input'
        self.correct_answer_col = None

    def transform_labels(self, annotations):
        labels = []
        if annotations:
            for annotation in annotations:
                print(annotation)
                if "human-atomic-facts" in annotation:
                    human_atomic_facts = annotation["human-atomic-facts"]
                    if human_atomic_facts:
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

    def fetch_wikidata(self, params):
        url = 'https://www.wikidata.org/w/api.php'
        try:
            return requests.get(url, params=params)
        except:
            return 'There was and error'

    def get_wiki_article(self, topic):
        # Which parameters to use
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'search': topic,
            'language': 'en'
        }
        # Fetch API
        data = self.fetch_wikidata(params)
        # show response as JSON
        data = data.json()
        try:
            id = data['search'][0]['id']
            params = {
                'action': 'wbgetentities',
                'ids': id,
                'format': 'json',
                'languages': 'en'
            }
            # fetch the API
            data = self.fetch_wikidata(params)
            # Show response
            data = data.json()
            url = wikipedia.page(data["entities"][id]['sitelinks']['enwiki']['title']).url
            return self.retrieve_wiki_article(url)

        except:
            try:
                article_name = wikipedia.search(topic)[0]
                url = wikipedia.page(article_name).url
                return self.retrieve_wiki_article(url)
            except:
                try:
                    url = f"https://en.wikipedia.org/wiki/{'_'.join(topic.split())}"
                    return self.retrieve_wiki_article(url)

                except:
                    print(f"Content not found on the page for {topic}")
                    return ""

    def load(self):
        self.load_jsonl_data()
        self.data["article"] = self.data["topic"].apply(lambda x: self.get_wiki_article(x))
        self.data["label"] = self.data["annotations"].apply(lambda x: self.transform_labels(x))


class ExpertQAData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = None
        self.generations_col = 'answer_string'
        self.labels_col = 'claims'
        self.query_col = 'question'
        self.correct_answer_col = None

    def load(self, model):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                temp = json.loads(line)
                try:
                    temp["answers"][model]["question"] = temp["question"]
                    data.append(temp["answers"][model])
                except KeyError:
                    print(f"KeyError for {model}")

        self.data = pd.DataFrame(data)

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

        # print(self.data.head())
        # print(self.data.columns)
        # print(self.data[['error_type', 'corrected_error_type']])


class FAVAData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = None
        self.generations_col = 'output'
        self.labels_col = 'label'
        self.query_col = 'prompt'
        self.correct_answer_col = 'reference'

    def assign_label(self, input_string):
        # Define the pattern using regular expression to capture any text between < and >
        pattern = r'<([^<>]+)>(.*?)<\/\1>'

        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        # If a match is found, return 1; otherwise, return 0
        return 1 if match else 0

    def remove_error_types(self, input_text):
        # Define the regular expression pattern
        pattern = r'<([^>]+)>'

        # Use re.sub to replace all occurrences of the pattern with an empty string
        result_text = re.sub(pattern, '', input_text)

        return result_text

    def load(self, model):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data_temp = json.load(f)

        for line in data_temp:
            if line["model"] == model:
                data.append(line)

        self.data = pd.DataFrame(data)
        self.data["reference"] = self.data["annotated"].apply(lambda x: self.remove_error_types(x))
        self.data["label"] = self.data["annotated"].apply(lambda x: self.assign_label(x))

        # print(self.data.head())
        # print(self.data.columns)

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
                #print(f"Article: {one_line['article']}")
                for sentence in one_line["summary_sentences"]:
                    #print(f"Sentence: {sentence['sentence']}")
                    #print(f"Responses: {sentence['responses']}")
                    article["generated_summary"] += sentence["sentence"]+" "
                    sentence["label"] = self.most_frequent_response(sentence["responses"])
                    labels_in_article.append(sentence["label"])
                    sentence["article"] = one_line["article"]
                    data_sentences.append(sentence)
                article["label"] = "no" if "no" in labels_in_article else "yes"
                data_articles.append(article)

                # print(len(data_sentences))
                # print(len(data_articles))

        if granularity == "sentence":
            self.data = pd.DataFrame(data_sentences)
        elif granularity == "article":
            self.data = pd.DataFrame(data_articles)
        else:
            raise ValueError("Unsupported granularity. Expecting 'sentence' or 'article'.")


        # print(self.data["label"].value_counts())
        # print(self.data)
        # print(self.data.columns)


class ScreenEvalData(ReadData):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.references_col = 'original_convo'
        self.generations_col = 'inferred_summary'
        self.labels_col = 'agg_label'
        self.query_col = None
        self.correct_answer_col = None

    def transform_labels(self, label):
        if label:
            return 0
        else:
            return 1

    def load(self, model):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model_ids = []
        for id in data["summary_id"].keys():
            if data["summary_id"][id] == model:
                model_ids.append(id)
        print(len(model_ids))
        dict_data = {}
        for col in data.keys():
            dict_data[col] = [data[col][id] for id in model_ids]
        self.data = pd.DataFrame(dict_data)
        #self.data['annotated_summary'] = self.data["annotated_summary"].str.replace('<mark>', '').str.replace('</mark>', '')
        self.data["agg_label"] = self.data["agg_label"].apply(lambda x: self.transform_labels(x))



# Example usage:
if __name__ == "__main__":
    for model in ["longformer", "gpt4", "human"]:
        data = ScreenEvalData("data/screen_eval.json")
        data.load(model)
        output = data.read_data()
        for ind in range(3):
            for col in output.columns:
                print(f"{col}: {output.loc[ind, col]}")
        print(output.columns)
        print(output)
