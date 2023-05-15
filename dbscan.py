import numpy as np

def dbscanFunction(data_points, eps, min_samples):
    # Compute pairwise distance matrix
    dist_mat = np.zeros((len(data_points), len(data_points)))
    for i in range(len(data_points)):
        for j in range(len(data_points)):
            dist_mat[i, j] = np.linalg.norm(data_points[i] - data_points[j])

    # Find core points
    core_points = []
    for i in range(len(data_points)):
        if np.sum(dist_mat[i] <= eps) >= min_samples:
            core_points.append(i)

    # Find border points
    border_points = []
    for i in range(len(data_points)):
        if i not in core_points and np.sum(dist_mat[i] <= eps) > 0:
            border_points.append(i)

    # Find noise points / outliers
    noise_points = []
    for i in range(len(data_points)):
        if i not in core_points and i not in border_points:
            noise_points.append(i)

    # Assign labels to core points and their neighbors
    labels = np.full(len(data_points), -1)
    cluster_id = 0
    for i in core_points:
        if labels[i] == -1:
            labels[i] = cluster_id
            neighbors = [j for j in range(len(data_points)) if dist_mat[i, j] <= eps]
            for j in neighbors:
                if labels[j] == -1 or labels[j] == 0:
                    labels[j] = cluster_id
            cluster_id += 1

    # Assign labels to border points
    for i in border_points:
        neighbors = [j for j in range(len(data_points)) if dist_mat[i, j] <= eps]
        neighbor_labels = [labels[j] for j in neighbors]
        unique_labels = set(neighbor_labels)
        if len(unique_labels) == 1:
            labels[i] = neighbor_labels[0]

    return labels
