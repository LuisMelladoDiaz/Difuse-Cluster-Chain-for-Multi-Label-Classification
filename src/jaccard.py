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

def plot_jaccard_scores_ECC(chain_jaccard_scores, ensemble_jaccard_score, num_chains):
    """
    Plot a comparison of Jaccard similarity scores for classifier chains and their ensemble.

    Args:
        chain_jaccard_scores: List of Jaccard scores for each classifier chain.
        ensemble_jaccard_score: Jaccard score for the ensemble of classifier chains.
        num_chains: Number of classifier chains used in the ensemble.
    """
    model_names = [f"Chain {i+1}" for i in range(num_chains)] + ["Ensemble"]
    model_scores = chain_jaccard_scores + [ensemble_jaccard_score]

    assert len(model_names) == len(model_scores), "The number of model names and scores must match."

    x_pos = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.grid(True)
    ax.set_title("Classifier Chain Ensemble Performance Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation="vertical")
    ax.set_ylabel("Jaccard Similarity Score")
    ax.set_ylim([min(model_scores) * 0.9, max(model_scores) * 1.1])
    colors = ["b"] * len(chain_jaccard_scores) + ["g"]
    ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
    plt.tight_layout()
    plt.show()
