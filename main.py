from read_load_data import *

# loading all the data

def load_data_by_name(dataset_name):
    dataset_output = {}
    dataset_name = dataset_name.lower()
    if dataset_name == "selfcheckgpt":
        data = SelfCheckGPTData("potsawee/wiki_bio_gpt3_hallucination")
        data.load()
        output = data.read_data()
        dataset_output["SelfCheckGPT"] = output
    elif dataset_name == "phd":
        for subset in ['wiki_1000w', 'wiki_10w', 'wiki_1y']:
            print(f"PHD_{subset}")
            data = PHDData("data/PHD_benchmark.json")
            data.load(subset)
            output = data.read_data()
            dataset_output[f"PHD_{subset}"] = output
    elif dataset_name == "halueval":
        for path_to_dataset in Path("data/HaluEval").glob("*"):
            print(f"HaluEval_{path_to_dataset.stem}")
            data = HaluEvalData(path_to_dataset)
            data.load()
            output = data.read_data()
            dataset_output[f"HaluEval_{path_to_dataset.stem}"] = output
    elif dataset_name == "bamboo":
        for path_to_dataset in Path("data/BAMBOO").glob("*"):
            print(f"BAMBOO_{path_to_dataset.stem}")
            data = BAMBOOData(path_to_dataset)
            data.load()
            output = data.read_data()
            dataset_output[f"BAMBOO_{path_to_dataset.stem}"] = output
    elif dataset_name == "felm":
        for subset in ["math", "reasoning", "science", "wk", "writing_rec"]:
            print(f"FELM_{subset}")
            data = FELMData("hkust-nlp/felm")
            data.load(subset)
            output = data.read_data()
            dataset_output[f"FELM_{subset}"] = output
    elif dataset_name == "factscore":
        for path_to_dataset in Path("data/FactScore").glob("*.jsonl"):
            print(f"FactScore_{path_to_dataset.stem}")
            data = FactScoreData(path_to_dataset)
            data.load()
            output = data.read_data()
            dataset_output[f"FactScore_{path_to_dataset.stem}"] = output
    elif dataset_name == "expertqa":
        for model in ["gpt4", "bing_chat", "rr_sphere_gpt4", "rr_gs_gpt4", "post_hoc_sphere_gpt4", "post_hoc_gs_gpt4"]:
            print(f"ExpertQA_{model}")
            data = ExpertQAData("data/ExpertQA/r2_compiled_anon.jsonl")
            data.load(model)
            output = data.read_data()
            dataset_output[f"ExpertQA_{model}"] = output
    elif dataset_name == "fava":
        for model in ['chatgpt', 'llama']:
            print(f"FAVA_{model}")
            data = FAVAData("data/fava.json")
            data.load(model)
            output = data.read_data()
            dataset_output[f"FAVA_{model}"] = output
    elif dataset_name == "factool":
        for path_to_dataset in Path("data/FacTool").glob("*.jsonl"):
            print(f"FacTool_{path_to_dataset.stem}")
            data = FacToolData(path_to_dataset)
            data.load()
            output = data.read_data()
            dataset_output[f"FacTool_{path_to_dataset.stem}"] = output
    elif dataset_name == "bump":
        for path_to_dataset in Path("data/BUMP").glob("*.json"):
            print(f"BUMP_{path_to_dataset.stem}")
            data = BUMPData(path_to_dataset)
            data.load()
            output = data.read_data()
            dataset_output[f"BUMP_{path_to_dataset.stem}"] = output
    else:
        raise ValueError("Invalid dataset name. Please choose from the following: SelfCheckGPT, PHD, HaluEval, BAMBOO, FELM, FactScore, ExpertQA, FAVA, FacTool")
    return dataset_output

if __name__ == "__main__":
    for dataset_name in ["BUMP", "ExpertQA"]: #"SelfCheckGPT", "PHD", "HaluEval", "BAMBOO", "FELM", "FactScore", "FAVA", "FacTool", "BUMP",
        dataset_output = load_data_by_name(dataset_name)
        print("-"*50)
