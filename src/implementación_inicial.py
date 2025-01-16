import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzy_c_means import fuzzy_c_means
from load_dataset import load_arff_data

def implementacion_inicial(filename, q_labels, n_clusters, m=2):
    _, _, _, labels = load_arff_data(filename, q_labels, sparse=False, return_labels_and_features=True)
    label_names, label_vectors = extraer_etiquetas(labels)
    clusters, centroids, U = fuzzy_c_means(label_vectors, n_clusters, m)
    plot_clusters_labels(centroids, label_names, clusters, U)


def extraer_etiquetas(labels):
    label_names = [label[0] for label in labels]
    label_vectors = np.eye(len(label_names))
    return label_names, label_vectors


def plot_clusters_labels(centroids, label_names, clusters, U):
    plt.figure(figsize=(10, 8))
    label_vectors = np.eye(len(label_names))[:, :2]
    centroids = centroids[:, :2]
    palette = sns.color_palette('viridis', len(set(clusters)))
    colors = [palette[cluster] for cluster in clusters]
    for i, (x, y) in enumerate(label_vectors):
        plt.scatter(x, y, s=200, c=[colors[i]], label=label_names[i])
        plt.annotate(label_names[i], (x + 0.02, y + 0.02), fontsize=10, weight='bold', rotation=45)
    for i, (cx, cy) in enumerate(centroids):
        plt.scatter(cx, cy, c=[palette[i]], marker='X', s=300, edgecolors='black', linewidth=2, label=f'Centroid {i}')
    plt.title('Clustering Difuso de Etiquetas')
    plt.tight_layout()
    col_labels = [f'Cluster {i}' for i in range(len(centroids))]
    row_labels = label_names
    plt.table(cellText=U.round(2), rowLabels=row_labels, colLabels=col_labels, loc='bottom', bbox=[0.2, -0.5, 0.6, 0.4])
    plt.subplots_adjust(bottom=0.4)
    plt.show()
