from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from utils.plotting import plot_dendrogram
from utils.preprocessing import load_arff_data
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

from utils.similarity import compute_similarity_matrix, convert_to_dissimilarity_matrix

## PREPROCESSING ##############################################################################################################################################################################################

def load_and_preprocess_data(file_path, num_labels, sparse):
    """Loads and preprocesses the multi-label dataset."""

    X, Y, features, labels = load_arff_data(
        filename=file_path,
        label_count=num_labels,
        sparse=sparse,
        return_labels_and_features=True
    )
    label_names = [label[0] for label in labels]

    return X, Y, features, label_names

## LABEL CLUSTER ##############################################################################################################################################################################################

def hierarchical_clustering(dissimilarity_matrix, method='ward'):
    """Performs hierarchical clustering using the Ward method."""

    compressed_distances = dissimilarity_matrix[np.triu_indices_from(dissimilarity_matrix, k=1)]
    linkage_matrix = linkage(compressed_distances, method=method)
    return linkage_matrix


def select_optimal_partition(linkage_matrix, dissimilarity_matrix, max_clusters, label_names):
    """Selects the optimal label clustering partition based on silhouette score."""

    best_score = -1
    best_partition = None
    best_num_clusters = 0

    for num_clusters in range(2, max_clusters + 1):
        
        partition = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        score = silhouette_score(dissimilarity_matrix, partition, metric="precomputed")

        if score > best_score:
            best_score = score
            best_partition = partition
            best_num_clusters = num_clusters

    cluster_dict = {}
    for label, cluster in zip(label_names, best_partition):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(label)

    for cluster_id, labels in sorted(cluster_dict.items()):
        print(f"Cluster {cluster_id}: {', '.join(labels)}")

    return best_partition


## LABEL CLUSTER CHAIN ##############################################################################################################################################################################################

def train_classifiers_per_cluster(X_train, Y_train, best_partition):
    """Trains multi-label classifiers for each label cluster."""

    models = []
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train_enhanced = X_train.copy()

    for cluster_id in np.unique(best_partition):
        indices_cluster = np.where(best_partition == cluster_id)[0]
        Y_train_cluster = Y_train[:, indices_cluster]
        base_model = LogisticRegression(max_iter=1000)
        cluster_model = MultiOutputClassifier(base_model)
        cluster_model.fit(X_train_enhanced, Y_train_cluster)
        Y_pred_cluster = cluster_model.predict(X_train_enhanced)
        X_train_enhanced = np.hstack((X_train_enhanced, Y_pred_cluster))
        models.append(cluster_model)

    return models

def predict_and_combine(models, X_test, best_partition, num_labels):
    """Performs prediction using trained cluster models and propagates predictions as new features."""

    Y_pred_final = np.zeros((X_test.shape[0], num_labels))
    X_test_enhanced = X_test.copy()

    for cluster_id, model in enumerate(models):

        indices_cluster = np.where(best_partition == cluster_id + 1)[0]
        Y_pred_cluster = model.predict(X_test_enhanced)
        X_test_enhanced = np.hstack((X_test_enhanced, Y_pred_cluster))
        Y_pred_final[:, indices_cluster] = Y_pred_cluster

    return Y_pred_final


## LABEL CLUSTER CHAIN FOR MULTILABEL CLASSIFICATION ##############################################################################################################################################################################################

def LCC_MLC(file_path, num_labels, sparse=False, max_clusters=10):
    """
    Implements the Label Cluster Chains for Multi-Label Classification (LCC-MLC) method.
    
    Steps:
    1. Load and preprocess the dataset.
    2. Compute label similarity and dissimilarity matrices.
    3. Perform hierarchical clustering to identify label clusters.
    4. Select the best partition using silhouette score.
    5. Train multi-label classifiers for each label cluster.
    6. Predict using trained models and combine predictions.
    
    Returns:
    - Final predicted labels for the dataset.
    """

    X, Y, features, label_names = load_and_preprocess_data(file_path, num_labels, sparse)
    similarity_matrix = compute_similarity_matrix(Y)
    dissimilarity_matrix = convert_to_dissimilarity_matrix(similarity_matrix)
    linkage_matrix = hierarchical_clustering(dissimilarity_matrix)
    plot_dendrogram(linkage_matrix, labels=label_names)
    best_partition = select_optimal_partition(linkage_matrix, dissimilarity_matrix, max_clusters, label_names)
    models = train_classifiers_per_cluster(X, Y, best_partition)
    Y_pred_final = predict_and_combine(models, X, best_partition, num_labels)

    print("Final predictions for all instances:")
    print(Y_pred_final)
    return Y_pred_final