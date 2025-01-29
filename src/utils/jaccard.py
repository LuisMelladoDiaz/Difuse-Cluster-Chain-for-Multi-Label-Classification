from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import numpy as np

def compute_jaccard_similarity(y_true, y_pred):
    """
    Compute the Jaccard similarity score for multi-label classification.

    Args:
        y_true: Ground truth labels (binary multi-label array).
        y_pred: Predicted labels (binary multi-label array).

    Returns:
        Jaccard similarity score averaged across all samples.
    """
    return jaccard_score(y_true, y_pred, average="samples", zero_division=0)

def compute_jaccard_index(y_vector_1, y_vector_2):
    """
    Compute the Jaccard similarity index between two binary label vectors.

    Args:
        y_vector_1: First binary label vector.
        y_vector_2: Second binary label vector.

    Returns:
        Jaccard similarity index between the two vectors.
    """
    intersection = np.sum((y_vector_1 == 1) & (y_vector_2 == 1))
    union = np.sum((y_vector_1 == 1) | (y_vector_2 == 1))
    return intersection / union if union > 0 else 0.0

