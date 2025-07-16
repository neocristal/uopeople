# ========================================
# WINE CLUSTERING ANALYSIS - Full Pipeline
# Save this code in a .txt file for submission
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# -------------------------------
# LOAD AND FIX THE DATASET
# -------------------------------

# Read the raw CSV file
raw = pd.read_csv("WineQT_unlabelled.csv", header=None)

# Check if file has only 1 giant column
if raw.shape[1] == 1:
    # Split by commas
    data = raw[0].str.split(",", expand=True)

    # Set first row as header
    data.columns = data.iloc[0]
    data = data.drop(0).reset_index(drop=True)

    # Convert numeric columns from strings
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="ignore")
else:
    # If file loads properly (multiple columns already)
    data = pd.read_csv("WineQT_unlabelled.csv")

print("Columns in CSV after cleaning:")
print(data.columns)

# -------------------------------
# PREPROCESSING
# -------------------------------

# These are the columns you want to cluster on
features = ["pH", "alcohol", "total sulfur dioxide"]

# Check missing columns
missing_cols = [col for col in features if col not in data.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

# Select desired features
X = data[features]

# Check for missing values
print("Missing values in selected features:")
print(X.isnull().sum())

# Drop rows with any missing data
X = X.dropna()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# CLUSTERING ALGORITHM
# -------------------------------

# Elbow method
inertia_list = []
k_values = range(1, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_list.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(6,4))
plt.plot(k_values, inertia_list, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid()
plt.tight_layout()
plt.savefig('elbow_plot.jpg')
plt.show()

# Based on elbow, choose k=3
optimal_k = 3

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

# Add cluster labels back to DataFrame
data['Cluster'] = clusters

# Silhouette score
sil_score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score for k={optimal_k}: {sil_score:.4f}")

# -------------------------------
# VISUALIZATION
# -------------------------------

# Scatter plot of clusters
sns.set(style="whitegrid")

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='alcohol',
    y='total sulfur dioxide',
    hue='Cluster',
    data=data,
    palette='Set2'
)
plt.title('Wine Clusters based on Alcohol and Total Sulfur Dioxide')
plt.xlabel('Alcohol Content')
plt.ylabel('Total Sulfur Dioxide')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('wine_clusters.jpg')
plt.show()

# Optional pairplot
sns.pairplot(
    data,
    vars=["pH", "alcohol", "total sulfur dioxide"],
    hue="Cluster",
    palette="Set2",
    plot_kws={'alpha':0.6}
)
plt.savefig('wine_clusters_pairplot.jpg')
plt.show()

# Cluster centers in original scale
cluster_centers = scaler.inverse_transform(kmeans_final.cluster_centers_)
cluster_summary = pd.DataFrame(cluster_centers, columns=features)

print("\nCluster Centers (original scale):")
print(cluster_summary)
