import os

import pandas as pd

def join_with_extended():
    for score in ["AlignScorer","ScaleScorer"]:
        for dataset in ["HaluEval_summarization_data", "HaluEval_dialogue_data", "HaluEval_qa_data"]: #"FAVA_chatgpt", "FAVA_llama",
            print(dataset)
            path_to_correct = "_".join([dataset.split("_")[0], "correct", "_".join(dataset.split("_")[1:])])
            path_to_negative = os.path.join("outputs", dataset, f"{score}_updated_data.csv")
            path_to_positive = os.path.join("outputs", path_to_correct, f"{score}_updated_data_add.csv")

            df_negative = pd.read_csv(path_to_negative, encoding="utf-8")
            df_positive = pd.read_csv(path_to_positive, encoding="utf-8")
            if "summarization" in dataset:
                df_negative = df_negative.dropna(subset=["generations"])
                df_positive = df_positive.dropna(subset=["generations"])
                df_negative = df_negative.drop_duplicates(subset=["generations"], keep="first")
                df_positive = df_positive.drop_duplicates(subset=["generations"], keep="first")
            else:
                df_negative = df_negative.dropna(subset=["query"])
                df_positive = df_positive.dropna(subset=["query"])
                df_negative = df_negative.drop_duplicates(subset=["query"], keep="first")
                df_positive = df_positive.drop_duplicates(subset=["query"], keep="first")

            print(df_negative.shape, df_positive.shape)
            df_updated = pd.concat([df_negative, df_positive], ignore_index=True)
            print(df_updated.shape)

            df_updated.to_csv(os.path.join("outputs", dataset, f"{score}_updated_data_new.csv"), encoding="utf-8", index=False)


join_with_extended()