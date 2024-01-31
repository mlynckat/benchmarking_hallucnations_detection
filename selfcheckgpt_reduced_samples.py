import json
import os

import pandas as pd

from methods_implementations import SelfCheckGPT

# read pandas dataframe from csv

path_to_df = "outputs/SelfCheckGPT/SelfCheckGPT_updated_data.csv"

df = pd.read_csv(path_to_df)
print(f"The size of the dataset is {df.shape[0]}")
# drop duplicates
df = df.drop_duplicates(subset=['query'], keep='first')
print(f"The size of the dataset is {df.shape[0]}")
# drop rows with nan values
df = df.drop(['0'], axis=1)
df = df.dropna(subset=['query'])

print(f"The size of the dataset is {df.shape[0]}")

# load config.json
with open("config.json") as json_file:
    config = json.load(json_file)

def apply_method(row, method, benchmark):
    query = config[benchmark]['prompt'].format(query=row['query'])
    model_name = config[benchmark]['model_name']
    predictions = method.make_predictions_no_sampling(row, query=query, model_name=model_name)
    return predictions

selfcheck = SelfCheckGPT()

if os.path.exists(os.path.join("outputs/SelfCheckGPT", f"{selfcheck.__class__.__name__}_updated_data_reduced.csv")):
    updated_data = pd.read_csv(
        os.path.join(os.path.join("outputs/SelfCheckGPT", f"{selfcheck.__class__.__name__}_updated_data_reduced.csv")),
        encoding="utf-8")
    updated_data_exists = True
    starting_point = updated_data.shape[0]
else:
    starting_point = 0
    updated_data_exists = False

for step in range(starting_point, df.shape[0], 5):
    end_index = min(step + 5, df.shape[0] - 1)
    step_data = df.loc[step+1:end_index].apply(lambda row: apply_method(row, selfcheck, "SelfCheckGPT"), axis=1)
    # concat to th updated_data if exists
    if updated_data_exists:
        updated_data = pd.concat([updated_data, step_data], ignore_index=True)
    else:
        updated_data = step_data
        updated_data_exists = True


    updated_data.to_csv(os.path.join("outputs/SelfCheckGPT", f"{selfcheck.__class__.__name__}_updated_data_reduced.csv"),
                                        encoding="utf-8", index=False)