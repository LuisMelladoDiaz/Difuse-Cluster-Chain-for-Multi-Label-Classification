import numpy as np
import warnings
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from evaluation_metrics.multilabel_evaluation import evaluate_multilabel_classification
from utils.similarity import compute_cluster_similarity, compute_label_similarity_matrix, convert_to_dissimilarity_matrix

### ========================== UTILIDADES GENERALES ==========================

def append_predictions_to_features(features, predictions):
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    return np.hstack((features, predictions))

def train_cluster_model(X, y, label_indices, random_state=None):
    y_cluster = y[:, label_indices]
    model = RandomForestClassifier(n_estimators=10, random_state=random_state, n_jobs=-1)
    model.fit(X, y_cluster)
    return model

def predict_cluster(model, X, label_indices, y_pred):
    cluster_pred = model.predict(X)
    
    if cluster_pred.ndim == 1:
        cluster_pred = cluster_pred.reshape(-1, 1)

    for j, label_idx in enumerate(label_indices):
        y_pred[:, label_idx] = cluster_pred[:, j]
    
    return cluster_pred

### ========================== CLUSTERING Y ORDENAMIENTO ==========================

def find_optimal_clusters(similarity_matrix, max_clusters=10):
    dissimilarity_matrix = convert_to_dissimilarity_matrix(similarity_matrix)
    distances = squareform(dissimilarity_matrix)
    linkage_matrix = linkage(distances, method='ward')

    best_score = -1
    best_labels = None
    best_n_clusters = 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n_clusters in range(2, min(max_clusters + 1, similarity_matrix.shape[0])):
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            try:
                score = silhouette_score(dissimilarity_matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_labels = labels
            except:
                continue

    return best_labels, best_n_clusters

def order_clusters(cluster_labels, similarity_matrix):
    cluster_similarities = compute_cluster_similarity(cluster_labels, similarity_matrix)
    n_clusters = cluster_similarities.shape[0]

    ordered_clusters = []
    remaining = list(range(n_clusters))

    current = np.argmin(np.mean(cluster_similarities, axis=1))
    ordered_clusters.append(current)
    remaining.remove(current)

    while remaining:
        similarities = [cluster_similarities[current, j] for j in remaining]
        current = remaining[np.argmax(similarities)]
        ordered_clusters.append(current)
        remaining.remove(current)

    return [x + 1 for x in ordered_clusters]

### ========================== ENTRENAMIENTO ==========================

def train_LCC_MLC(X_train, y_train, random_state=None):
    similarity_matrix = compute_label_similarity_matrix(y_train)
    cluster_labels, n_clusters = find_optimal_clusters(similarity_matrix)
    cluster_order = order_clusters(cluster_labels, similarity_matrix)

    models = []
    cluster_label_mapping = []
    current_features = X_train.copy()

    for cluster_idx in cluster_order:
        mask = cluster_labels == cluster_idx
        label_indices = np.where(mask)[0]
        cluster_label_mapping.append(label_indices)

        model = train_cluster_model(current_features, y_train, label_indices, random_state)
        models.append(model)

        if len(models) < n_clusters:
            predictions = model.predict(current_features)
            current_features = append_predictions_to_features(current_features, predictions)

    return {
        'models': models,
        'cluster_label_mapping': cluster_label_mapping,
        'n_clusters': n_clusters,
        'cluster_order': cluster_order,
        'cluster_labels': cluster_labels
    }

### ========================== PREDICCIÓN ==========================

def predict_LCC_MLC(X, trained_model):
    models = trained_model['models']
    cluster_label_mapping = trained_model['cluster_label_mapping']
    n_clusters = trained_model['n_clusters']

    n_labels = sum(len(mapping) for mapping in cluster_label_mapping)
    y_pred = np.zeros((X.shape[0], n_labels))
    current_features = X.copy()

    for i, (model, label_indices) in enumerate(zip(models, cluster_label_mapping)):
        cluster_pred = predict_cluster(model, current_features, label_indices, y_pred)

        if i < n_clusters - 1:
            current_features = append_predictions_to_features(current_features, cluster_pred)

    return y_pred

### ========================== PIPELINE COMPLETO ==========================

def LCC_MLC(X_train, y_train, X_test, y_test, random_state=None):

    trained_model = train_LCC_MLC(X_train, y_train, random_state)
    y_pred = predict_LCC_MLC(X_test, trained_model)

    metrics = evaluate_multilabel_classification(y_test, y_pred)

    return y_pred, metrics