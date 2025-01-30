
import numpy as np

from utils.jaccard import compute_jaccard_index


def compute_similarity_matrix(Y):
    """Computes the label similarity matrix using the Jaccard index."""

    num_labels = Y.shape[1]
    similarity_matrix = np.zeros((num_labels, num_labels))

    for i in range(num_labels):
        for j in range(num_labels):

            similarity_matrix[i, j] = compute_jaccard_index(Y[:, i], Y[:, j])

    return similarity_matrix

def convert_to_dissimilarity_matrix(similarity_matrix):
    """Converts the similarity matrix into a dissimilarity matrix."""

    return 1 - similarity_matrix
