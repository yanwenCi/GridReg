import numpy as np
from sklearn.cluster import KMeans

# Generate some sample data (replace this with your actual data)
np.random.seed(42)
data = np.random.rand(100, 2)  # Sample 100 data points in 2D

# Number of clusters (just 1 for computing the mean)
num_clusters = 2

# Apply K-means clustering with one cluster
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data)

# Get the cluster centroids (in this case, there will be only one centroid)
centroid = kmeans.cluster_centers_

# Compute the mean of the data directly
mean_data = np.mean(data, axis=0)

# Display the centroid obtained from K-means and the mean of the data
print("Centroid from K-means (Mean of Data Points):", np.mean(centroid, 0))
print("Mean of the Data Points:", mean_data)
