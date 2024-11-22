import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = 'Result_CN Analisis Samples Robinia from Ukraine.xlsx'
data = pd.ExcelFile(file_path)

# Load the 'N' sheet and ignore the first column
data_n = data.parse('N').iloc[:, 1:]

# Rename the columns for consistency
data_n.columns = data_n.columns.str.strip()

# Add a column to group data into bins of 10 samples
data_n['Sample Group'] = (data_n.index // 10) + 1

# Filter the dataset into only the first 16 groups
filtered_data_n = data_n[data_n['Sample Group'] <= 16]

# Set up the plot for 16 sample groups (10 samples per group)
plt.figure(figsize=(20, 15))
plot_index = 1

# Group by 'Sample Group'
sample_group_categories = filtered_data_n.groupby(['Sample Group'])

# Generate box plots for each group
for sample_group, group in sample_group_categories:
    if plot_index > 16:  # Limit to 16 plots
        break
    plt.subplot(4, 4, plot_index)
    sns.boxplot(data=group, x='Leaves Category', y='N, %')
    plt.title(f"Sample Group {sample_group}")
    plt.xlabel("Leaves Category")
    plt.ylabel("N, %")
    plot_index += 1

plt.tight_layout()
plt.show()
