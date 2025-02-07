import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc



def plot_dendrogram(linkage_matrix, labels):
    """Plots the hierarchical clustering dendrogram."""
    plt.figure(figsize=(10, 10))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=10)
    plt.title("Dendrogram")
    plt.xlabel("Labels")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

def plot_clusters(data, labels, x_label = 'x', y_label = 'y'):
    """Plot the clustered data points with different colors based on labels."""
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='nipy_spectral')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


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

def plot_roc_curve(y_true, y_probs):
    """Plots the ROC curve for a binary classifier with probabilistic outputs."""
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    
    return auc_score