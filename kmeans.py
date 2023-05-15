import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Function to run K-means clustering
def k_means(data_points, num_clusters, max_iterations):
    # Set the dimensions of the feature space
    num_features = data_points.shape[1]

    # Generate random initial centroids
    centroids = np.zeros((num_clusters, num_features))
    min_vals = np.min(data_points, axis=0)
    max_vals = np.max(data_points, axis=0)
    for i in range(num_clusters):
        centroids[i, :] = np.random.uniform(min_vals, max_vals)

    # Iterate until convergence or maximum number of iterations is reached
    for iteration in range(max_iterations):
        # Assign each data point to the closest centroid
        distances = np.zeros((data_points.shape[0], num_clusters))
        for i in range(num_clusters):
            # Calculate the Euclidean distance from each data point to the centroid
            distances[:, i] = np.linalg.norm(data_points - centroids[i, :], axis=1)
        # Assign each data point to the closest centroid
        closest = np.argmin(distances, axis=1)

        # Update the centroids as the mean of the assigned data points
        for i in range(num_clusters):
            centroids[i, :] = np.mean(data_points[closest == i, :], axis=0)

    # Return the final centroids and the assignments of the data points
    return centroids, closest
