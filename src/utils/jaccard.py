from sklearn.metrics import jaccard_score
import numpy as np

def jaccard_similarity_score(y_true, y_pred):
    return jaccard_score(y_true, y_pred, average="samples", zero_division=0)

def jaccard_index(y1, y2):
    return _jaccard(y1, y2)

def jaccard_distance(y1, y2):
    return 1 - _jaccard(y1, y2)

def _jaccard(y1, y2):
    intersection = np.sum(np.logical_and(y1, y2))
    union = np.sum(np.logical_or(y1, y2))
    return intersection / union if union > 0 else 0.0
