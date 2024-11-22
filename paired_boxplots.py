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

# Rename 'N, %' to 'Metric' for consistency
data_n = data_n.rename(columns={'N, %': 'Metric'})

# Ensure 'Metric' is numeric, coercing errors to NaN and dropping invalid rows
data_n['Metric'] = pd.to_numeric(data_n['Metric'], errors='coerce')
data_n = data_n.dropna(subset=['Metric'])


def plot_selected_groups(groups_to_plot):
    """
    Generate a single plot for the specified sample groups.
    :param groups_to_plot: List of sample group numbers to include in the plot.
    """
    # Filter the dataset for the selected groups
    filtered_data = data_n[data_n['Sample Group'].isin(groups_to_plot)]

    # Plot the selected groups
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=filtered_data, x='Leaves Category', y='Metric', hue='Sample Group')
    plt.title(f"Sample Groups: {', '.join(map(str, groups_to_plot))}")
    plt.xlabel("Leaves Category")
    plt.ylabel("N, %")
    plt.legend(title="Sample Group")
    plt.tight_layout()
    plt.show()


# Example Usage:
# Input the desired groups to plot
groups = []  # Replace with your desired groups (e.g., [3, 4, 5])
plot_selected_groups(groups)
