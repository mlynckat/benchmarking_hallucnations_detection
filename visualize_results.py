#read json
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "outputs/results_full.json"

with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

# create empty dataframe
df = pd.DataFrame()

for dataset in data.keys():
    line = {}
    sum_hall_samples = 0
    len_hall_samples = 0
    for score in data[dataset].keys():
        line[score] = data[dataset][score]["geom_mean"]
        if data[dataset][score]["hall_samples"]:
            sum_hall_samples += data[dataset][score]["hall_samples"]
            len_hall_samples += 1
    line["hall_samples"] = sum_hall_samples / len_hall_samples
    curr_df = pd.DataFrame(line, index=[dataset])

    df = pd.concat([df, curr_df], axis=0)


f, axs = plt.subplots(1, 2, gridspec_kw={'wspace': 0, 'width_ratios': [1, 9]}, figsize=(13, 12))
sns.heatmap(df[["hall_samples"]], annot=True, fmt='.2f', ax=axs[0], cmap='Reds', cbar=False)

sns.heatmap(df.drop("hall_samples", axis=1), annot=True, cmap="YlGnBu", ax=axs[1])
axs[1].yaxis.set_ticks([])



plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/results_visualization_full.png")

df = pd.DataFrame()

for dataset in data.keys():
    line = {}
    sum_hall_samples = 0
    len_hall_samples = 0
    for score in data[dataset].keys():
        line[score] = f"{data[dataset][score]['precision']}/{data[dataset][score]['recall']}"
        if data[dataset][score]["hall_samples"]:
            sum_hall_samples += data[dataset][score]["hall_samples"]
            len_hall_samples += 1
    line["hall_samples"] = sum_hall_samples / len_hall_samples
    curr_df = pd.DataFrame(line, index=[dataset])

    df = pd.concat([df, curr_df], axis=0)


df.to_csv("outputs/final_table_full.csv", encoding="utf-8")

df = pd.DataFrame()

for dataset in data.keys():
    line = {}
    sum_hall_samples = 0
    len_hall_samples = 0
    for score in data[dataset].keys():
        line[score] = data[dataset][score]['f1_score']
        if data[dataset][score]["hall_samples"]:
            sum_hall_samples += data[dataset][score]["hall_samples"]
            len_hall_samples += 1
    line["hall_samples"] = sum_hall_samples / len_hall_samples
    curr_df = pd.DataFrame(line, index=[dataset])

    df = pd.concat([df, curr_df], axis=0)


df.to_csv("outputs/final_table_f1scores_full.csv", encoding="utf-8")


