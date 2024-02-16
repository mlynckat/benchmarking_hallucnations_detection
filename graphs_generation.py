import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the data
data = pd.read_csv('data/benchmarks_landscape.csv', encoding='utf-8', sep=";")  # Replace 'your_file.csv' with the actual filename

granularity_mapping = {"token": 0, "segment": 1, "sentence": 2, "passage": 3}
data['Granularity'] = data['Granularity'].map(granularity_mapping)

# Step 2: Set seaborn style
sns.set(style="ticks")

# Step 3: Plot a scatter plot
fig, ax = plt.subplots(figsize=(14, 8))
# Scatter plot with size dependent on the 'Size' column
sns.stripplot(data=data, x='Granularity', y='Size', hue='Task', palette='pastel', alpha=0.5, jitter=True, s=35)


# Create a new legend with modified spacing
plt.legend(labelspacing=1.3, fontsize=13, borderpad=1.5, markerscale = 0.7, title='Task')

plt.xlabel('Granularity', fontsize=16)
plt.ylabel('Dataset size', fontsize=16)

# Set x-axis ticks to the initial string values
ax.set_xticks([0, 1, 2, 3])
plt.yticks(fontsize=14)
ax.set_xticklabels(list(granularity_mapping.keys()), fontsize=16)

plt.tight_layout()

# Show the plot
plt.savefig('outputs/scatter_plot_benchmarks.png')