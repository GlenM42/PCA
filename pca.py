from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'table.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Encode categorical columns using one-hot encoding
encoded_data = pd.get_dummies(data, columns=["Location", "Tree type", "Tree age", "Leaves Category"])

# Separate the target column (N, %) and the features
features = encoded_data.drop(columns=["NumbersUA-sample", "N, % "])  # Handle column with trailing space
target = encoded_data["N, % "]  # Handle column with trailing space

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform PCA
pca = PCA()
pca_results = pca.fit_transform(scaled_features)

# Convert PCA results into a DataFrame
pca_df = pd.DataFrame(
    pca_results,
    columns=[f"PC{i+1}" for i in range(pca_results.shape[1])]
)

# Display explained variance for each component
explained_variance = pca.explained_variance_ratio_
explained_variance_df = pd.DataFrame({
    "Principal Component": [f"PC{i+1}" for i in range(len(explained_variance))],
    "Explained Variance Ratio": explained_variance
})

# Save PCA results to a CSV
output_path = 'pca_results.csv'  # Replace with your desired output path
pca_df.to_csv(output_path, index=False)
print(f"PCA results saved to {output_path}.")

# Scree plot for explained variance
plt.figure(figsize=(8, 6))
plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker='o',
    linestyle='--'
)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid()
plt.show()

# Scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(
    pca_df['PC1'],
    pca_df['PC2'],
    alpha=0.7
)
plt.title('PCA: First Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.show()

# Display explained variance for reference
print("Explained Variance by Component:")
print(explained_variance_df)

# Calculate loadings (correlations between original variables and PCs)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
    index=features.columns
)

# Prepare a summary for the loadings to understand variable contributions
loadings_summary = loadings.copy()

# Calculate the most important PC per variable
loadings_summary["Max PC Contribution"] = loadings.abs().idxmax(axis=1)

# Calculate the maximum absolute loading value
loadings_summary["Max Loading Value"] = loadings.abs().max(axis=1)

# Display the loadings summary
print("PCA Loadings with Variable Contributions:")
print(loadings_summary)
loadings_summary.to_csv('loadings_summary.csv')

# Loadings Matrix
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
    index=features.columns
)

# Group variables by their category
categories = {
    "Location": [col for col in loadings.index if "Location" in col],
    "Tree Type": [col for col in loadings.index if "Tree type" in col],
    "Tree Age": [col for col in loadings.index if "Tree age" in col],
    "Leaves Category": [col for col in loadings.index if "Leaves Category" in col]
}

# Calculate mean absolute loading for each category
category_importance = {}
for category, variables in categories.items():
    mean_abs_loading = loadings.loc[variables].abs().mean().mean()  # Mean across PCs
    category_importance[category] = mean_abs_loading

# Convert to DataFrame for better interpretation
importance_df = pd.DataFrame.from_dict(category_importance, orient='index', columns=["Mean Absolute Loading"])
importance_df = importance_df.sort_values(by="Mean Absolute Loading", ascending=False)

# Display category importance
print("Category Importance Based on PCA Loadings:")
print(importance_df)

# Scatter plot of the first two PCs
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, label='Observations')
plt.quiver(
    [0] * len(loadings.index),
    [0] * len(loadings.index),
    loadings['PC1'] * 5,  # Scale arrows for visibility
    loadings['PC2'] * 5,
    angles='xy', scale_units='xy', scale=1, color='r', alpha=0.75, label='Variables'
)
for i, var in enumerate(loadings.index):
    plt.text(loadings['PC1'][i] * 5, loadings['PC2'][i] * 5, var, color='r', fontsize=8)

plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.title("Biplot: PC1 vs. PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 3D Biplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the observations (data points)
ax.scatter(
    pca_df['PC1'],
    pca_df['PC2'],
    pca_df['PC3'],
    alpha=0.6, label='Observations'
)

# Add arrows for variable loadings
for i, var in enumerate(loadings.index):
    ax.quiver(
        0, 0, 0,
        loadings['PC1'][i] * 5,  # Scale arrows for visibility
        loadings['PC2'][i] * 5,
        loadings['PC3'][i] * 5,
        color='r', alpha=0.75
    )
    ax.text(
        loadings['PC1'][i] * 5,
        loadings['PC2'][i] * 5,
        loadings['PC3'][i] * 5,
        var, color='r', fontsize=8
    )

# Axes labels and title
ax.set_title("3D Biplot: PC1, PC2, PC3")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# Add grid and legend
ax.legend()
plt.tight_layout()
plt.show()
