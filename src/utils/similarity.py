import numpy as np
from utils.jaccard import jaccard_distance, jaccard_index

def compute_label_matrix(Y, similarity_fn, symmetric=True):

    num_labels = Y.shape[1]
    matrix = np.zeros((num_labels, num_labels))

    for i in range(num_labels):
        for j in range(i if symmetric else 0, num_labels):
            value = similarity_fn(Y[:, i], Y[:, j])
            matrix[i, j] = value
            if symmetric:
                matrix[j, i] = value

    return matrix

def compute_label_similarity_matrix(Y):
    return compute_label_matrix(Y, jaccard_index, symmetric=True)

def compute_label_distance_matrix(Y):
    return compute_label_matrix(Y, jaccard_distance, symmetric=True)

def convert_to_dissimilarity_matrix(similarity_matrix):
    return 1 - similarity_matrix

def compute_cluster_similarity(cluster_labels, similarity_matrix):
    n_clusters = len(np.unique(cluster_labels))
    cluster_sim = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            mask_i = cluster_labels == (i + 1)
            mask_j = cluster_labels == (j + 1)
            if np.any(mask_i) and np.any(mask_j):
                cluster_sim[i, j] = np.mean(similarity_matrix[mask_i][:, mask_j])

    return cluster_sim
