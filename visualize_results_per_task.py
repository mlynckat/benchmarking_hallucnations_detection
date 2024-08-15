#read json
import json
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


path = "outputs/results_full.json"
path_per_tak = "outputs/results_per_task.json"

with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

with open(path_per_tak, "r", encoding="utf-8") as file:
    data_per_task = json.load(file)

tasks = list(data_per_task.keys())
"""for task in tasks:
    for ind_metric, metric in enumerate(["precision", "recall", "f1_score", "geom_mean"]): #"accuracy",

        # create empty dataframe
        df = pd.DataFrame()
        scores = list(data_per_task[task].keys())
        datasets = list(data_per_task[task][scores[0]].keys())
        for dataset in datasets:
            if dataset != "best_threshold" and dataset != "best_gmean":
                line = {}
                sum_hall_samples = 0
                len_hall_samples = 0
                for score in scores:
                    line[score] = data[dataset][score][metric]
                    if data[dataset][score]["hall_samples"]:
                        sum_hall_samples += data[dataset][score]["hall_samples"]
                        len_hall_samples += 1
                line["hall_samples"] = sum_hall_samples / len_hall_samples
                curr_df = pd.DataFrame(line, index=[dataset])

                df = pd.concat([df, curr_df], axis=0)

        df_per_task = pd.DataFrame()
        for dataset in datasets:
            if dataset != "best_threshold" and dataset != "best_gmean":
                line = {}
                sum_hall_samples = 0
                len_hall_samples = 0
                for score in scores:
                    line[score] = round(data_per_task[task][score][dataset][metric], 2)
                    if data_per_task[task][score][dataset]["hall_samples"]:
                        sum_hall_samples +=data_per_task[task][score][dataset]["hall_samples"]
                        len_hall_samples += 1
                line["hall_samples"] = sum_hall_samples / len_hall_samples
                curr_df_per_task = pd.DataFrame(line, index=[dataset])

                df_per_task = pd.concat([df_per_task, curr_df_per_task], axis=0)


        f, axs = plt.subplots(1, 3, gridspec_kw={'wspace': 0.1, 'width_ratios': [1, 4.5, 4.5]}, figsize=(24, 12))

        with sns.axes_style('white'):

            sns.heatmap(df[["hall_samples"]], annot=True, fmt='.2f', ax=axs[0], cmap=ListedColormap(['white']), cbar=False, annot_kws={"size": 12})
            # place x ticks on top
            axs[0].xaxis.tick_top()
            axs[0].xaxis.set_label_position('top')


        sns.heatmap(df.drop("hall_samples", axis=1), annot=True, cmap="YlGnBu", ax=axs[1])


        sns.heatmap(df_per_task.drop("hall_samples", axis=1), annot=True, cmap="YlGnBu", ax=axs[2])

        axs[1].yaxis.set_ticks([])
        axs[2].yaxis.set_ticks([])

        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')
        axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45, ha='right')


        plt.savefig(f"outputs/results_visualization_{task}_{metric}.png", bbox_inches='tight')"""




for task in tasks:
    fig_task, axs_task = plt.subplots(2, 2, gridspec_kw={'wspace': 0.1, 'hspace': 0.4}, figsize=(16, 13))
    for ind_metric, metric in enumerate(["precision", "recall", "f1_score", "geom_mean"]): #"accuracy",

        # create empty dataframe
        df = pd.DataFrame()
        scores = list(data_per_task[task].keys())
        datasets = list(data_per_task[task][scores[0]].keys())
        for dataset in datasets:
            if dataset != "best_threshold" and dataset != "best_gmean":
                line = {}
                sum_hall_samples = 0
                len_hall_samples = 0
                for score in scores:
                    line[score] = data[dataset][score][metric]
                    if data[dataset][score]["hall_samples"]:
                        sum_hall_samples += data[dataset][score]["hall_samples"]
                        len_hall_samples += 1
                line["hall_samples"] = sum_hall_samples / len_hall_samples
                curr_df = pd.DataFrame(line, index=[dataset])

                df = pd.concat([df, curr_df], axis=0)

        df_per_task = pd.DataFrame()
        for dataset in datasets:
            if dataset != "best_threshold" and dataset != "best_gmean":
                line = {}
                sum_hall_samples = 0
                len_hall_samples = 0
                for score in scores:
                    line[score] = round(data_per_task[task][score][dataset][metric], 2)
                    if data_per_task[task][score][dataset]["hall_samples"]:
                        sum_hall_samples +=data_per_task[task][score][dataset]["hall_samples"]
                        len_hall_samples += 1
                line["hall_samples"] = sum_hall_samples / len_hall_samples
                curr_df_per_task = pd.DataFrame(line, index=[dataset])

                df_per_task = pd.concat([df_per_task, curr_df_per_task], axis=0)

        df1_melted = df.drop("hall_samples", axis=1).reset_index().melt(id_vars=['index'], var_name='Scorer', value_name='Score')
        df2_melted = df_per_task.drop("hall_samples", axis=1).reset_index().melt(id_vars=['index'], var_name='Scorer', value_name='Score')

        # Add a 'Source' column to distinguish between the two dataframes
        df1_melted['Source'] = f"dataset"
        df2_melted['Source'] = f"task"

        # Combine the melted dataframes
        df_combined = pd.concat([df1_melted, df2_melted], ignore_index=True)

        ax = axs_task[ind_metric // 2, ind_metric % 2]
        sns.boxplot(x='Scorer', y='Score', hue='Source', data=df_combined, palette='Set2', ax=ax)

        # Customize the subplot
        ax.set_title(f'{metric}', fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(title='Threshold optimized per:')
        scorer_names = df.columns  # Assuming both dataframes have the same column names
        ax.set_xticks(np.arange(len(scorer_names)))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xticklabels(scorer_names)

    fig_task.suptitle(f'Performance Metrics for {task}', fontsize=18, y=1.02)
    plt.savefig(f"outputs/results_boxplots_{task}.png", bbox_inches='tight')



for task in tasks:
    scores = list(data_per_task[task].keys())
    datasets = list(data_per_task[task][scores[0]].keys())
    df = pd.DataFrame()
    for score in scores:
        line = {}
        for metric in ["precision", "recall", "f1_score", "geom_mean"]:
            all_diffs = []
            for dataset in datasets:
                if dataset != "best_threshold" and dataset != "best_gmean":
                    try:
                        calc = data_per_task[task][score][dataset][metric] / data[dataset][score][metric]
                        all_diffs.append(calc)
                    except ZeroDivisionError:
                        pass

            #print(f"Task: {task}, Score: {score}, Metric: {metric}, Mean: {np.mean(all_diffs)}, Std: {np.std(all_diffs)}")

        # create a dataframe with columns for each metric and row for each mean score for all scores in one task
            line[metric] = np.mean(all_diffs)
        curr_df = pd.DataFrame(line, index=[score])
        df = pd.concat([df, curr_df], axis=0)
    print(df)
    # create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df, annot=True, ax=ax)
    ax.set_title(f"Task: {task}")
    plt.savefig(f"outputs/results_diffs_heatmap_{task}.png", bbox_inches='tight')





sys.exit()
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


df.to_csv("outputs/final_table.csv", encoding="utf-8")

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


df.to_csv("outputs/final_table_f1scores.csv", encoding="utf-8")


