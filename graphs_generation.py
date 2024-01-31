import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the data
data = pd.read_csv('data/benchmarks_landscape.csv', encoding='utf-8', sep=";")  # Replace 'your_file.csv' with the actual filename

# Step 2: Map the strings from the column Granularity to integers
granularity_mapping = {"token": 1, "segment": 2, "sentence": 3, "passage": 4}
data['Granularity'] = data['Granularity'].map(granularity_mapping)
plt.clf()
# Step 3: Plot a scatter plot
cmap = plt.cm.viridis
norm = plt.Normalize(0, len(data["Task"].unique()) - 1)
colors = cmap(norm(np.arange(len(data["Task"].unique()))))

fig, ax = plt.subplots()
print(data['Granularity'])
ax.scatter(data['Granularity'], data['Size'], c=data['Task'].astype('category').cat.codes, cmap='viridis', s=data['Size']*0.1, alpha=0)
ax.xaxis.set_ticks([ 1, 2, 3, 4 ])
ax.xaxis.set_ticklabels(list(granularity_mapping.keys()))
ax.set_xticks([ 1, 2, 3, 4 ], labels=list(granularity_mapping.keys()))



for i, task in enumerate(data["Task"].unique()):
    print(data[data["Task"] == task]['Granularity'])
    ax.scatter(list(data[data["Task"] == task]['Granularity']), list(data[data["Task"] == task]['Size']), color=colors[i], s=150,
                label=task, alpha=0.7)


# Step 6: Set x-axis ticks to the initial string values
plt.legend()
# Adding labels and title
plt.xlabel('Granularity')
plt.ylabel('Size')
plt.title('Scatter Plot of Data')

# Show the plot
plt.savefig('outputs/scatter_plot_benchmarks.png')