
import numpy as np
from utils.jaccard import compute_jaccard_distance

def compute_label_similarity_matrix(Y):
    """Calcula la matriz de similitud entre etiquetas utilizando el índice Jaccard."""
    num_labels = Y.shape[1]
    similarity_matrix = np.zeros((num_labels, num_labels))

    for i in range(num_labels):
        for j in range(num_labels):
            similarity_matrix[i, j] = compute_jaccard_distance(Y[:, i], Y[:, j])
    
    return similarity_matrix

def convert_to_dissimilarity_matrix(similarity_matrix):
    """Converts the similarity matrix into a dissimilarity matrix."""

    return 1 - similarity_matrix
