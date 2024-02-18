import ast
import os

import pandas as pd

for dataset in ["FAVA_chatgpt", "FAVA_llama", "FELM_math", "FELM_reasoning", "FELM_science", "FELM_wk", "FELM_writing_rec"]: #"FAVA_chatgpt", "FAVA_llama",
    print(dataset)
    path_to_sm = os.path.join("outputs", dataset, "SAC3_updated_data_sm.csv")
    path_to_extended = os.path.join("outputs", dataset, "SAC3_updated_data_extended.csv")

    df_sm = pd.read_csv(path_to_sm, encoding="utf-8")
    df_extended = pd.read_csv(path_to_extended, encoding="utf-8")
    #df_updated = df_extended.copy(deep=True)

    df_sm = df_sm.dropna(subset=["query"])
    df_extended = df_extended.dropna(subset=["query"])
    df_sm = df_sm.drop_duplicates(subset=["query"], keep="first")
    df_extended = df_extended.drop_duplicates(subset=["query"], keep="first")
    print(df_sm.shape, df_extended.shape)
    df_updated = pd.merge(df_sm, df_extended, on='query', sort=False)
    print(df_updated.columns)

    for col in ["sc2_vote", "sac3_q_vote", "sac3_qm(falcon)_vote", "sac3_qm(starling)_vote"]:
        df_updated[col+"_x"] = df_updated[col+"_x"].apply(ast.literal_eval)
        df_updated[col+"_y"] = df_updated[col+"_y"].apply(ast.literal_eval)
        df_updated["generations"] = df_updated["generations_y"]
        df_updated["labels"] = df_updated["labels_y"]


        # set "query" as index
        #df_sm.set_index("query", inplace=True)
        #df_extended.set_index("query", inplace=True)
        # find common indices
        #common_indices = df_sm.index.intersection(df_extended.index)
        # apply common indices to both df
        #df_sm = df_sm.loc[common_indices]
        #df_extended = df_extended.loc[common_indices]

        #assert df_sm.index.equals(df_extended.index), f"Indices of {path_to_sm} and {path_to_extended} do not match"
        df_updated[col+"_y"] = df_updated[col+"_y"].apply(lambda x: x[-7:])
        df_updated[col] = df_updated[col+"_x"] + df_updated[col+"_y"]
        df_updated[col + "_short"] = df_updated[col + "_y"].apply(lambda x: x[-3:])
        score_col = col.replace("vote", "score")
        cost_col = col.replace("vote", "cost")
        df_updated[score_col] = df_updated[col].apply(lambda x: sum(x) / len(x))
        df_updated[score_col + "_short"] = df_updated[col + "_short"].apply(lambda x: sum(x) / len(x))
        df_updated[cost_col] = df_updated[cost_col+"_x"] + df_updated[cost_col+"_y"]
    df_updated["starling_self"] = df_updated["sac3_qm(starling)_vote_y"].apply(lambda x: x[:7]) + df_updated["sac3_qm(starling)_vote_x"].apply(lambda x: x[:3])
    df_updated["sac3_vote(all)"] = df_updated["sac3_q_vote"] + df_updated["sac3_qm(starling)_vote"]
    df_updated["sac3_score(all)"] = df_updated["sac3_vote(all)"].apply(lambda x: sum(x) / len(x))
    df_updated["sac3_vote(all_no_perb)"] = df_updated["sac3_q_vote"] + df_updated["starling_self"]
    df_updated["sac3_score(all_no_perb)"] = df_updated["sac3_vote(all_no_perb)"].apply(lambda x: sum(x) / len(x))
    df_updated["sac3_vote(all)_short"] = df_updated["sac3_q_vote_short"] + df_updated["sac3_qm(starling)_vote_short"]
    df_updated["sac3_score(all)_short"] = df_updated["sac3_vote(all)_short"].apply(lambda x: sum(x) / len(x))

    df_updated.to_csv(os.path.join("outputs", dataset, "SAC3_updated_data.csv"), encoding="utf-8", index=False)