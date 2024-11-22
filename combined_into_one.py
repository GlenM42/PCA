import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

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


def plot_nested_groups_combined(nested_groups):
    """
    Combine data for nested groups and plot a single box plot on one axis system.
    :param nested_groups: List of lists of sample group numbers to combine.
    """
    # Combine all specified groups into one DataFrame
    combined_data = pd.concat([
        data_n[data_n['Sample Group'].isin(groups)] for groups in nested_groups
    ])

    # Add a column to label combined groups
    combined_data['Combined Groups'] = [
        f"Group {i + 1}" for i, groups in enumerate(nested_groups)
        for _ in range(len(data_n[data_n['Sample Group'].isin(groups)]))
    ]

    # Plot the combined data
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_data, x='Combined Groups', y='Metric')
    plt.title(f"Boxplot for Combined Groups: {', '.join(str(groups) for groups in nested_groups)}")
    plt.xlabel("Combined Groups")
    plt.ylabel("N, % (Metric)")
    plt.tight_layout()
    plt.show()


# Example Usage:
# Input the nested groups to combine
nested_groups = [[1, 3, 6, 9, 12, 16], [2, 4, 5, 7, 8, 10, 11, 13, 15]]  # Replace with your desired nested groups
plot_nested_groups_combined(nested_groups)
