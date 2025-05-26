import os
import numpy as np
import skfuzzy as fuzz
import random
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from evaluation_metrics.multilabel_evaluation import evaluate_multilabel_classification
from utils.similarity import compute_fuzzy_cluster_similarity, compute_label_distance_matrix

## ============================ FUZZY CLUSTERING =====================================

def apply_fuzzy_cmeans(similarity_matrix, num_clusters, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    max_distance = np.max(similarity_matrix)
    normalized_similarity_matrix = 1 - (similarity_matrix / max_distance)

    cntr, u, _, _, _, _, _ = fuzz.cmeans(normalized_similarity_matrix, num_clusters, 2.5, error=0.005, maxiter=1000, seed=seed)
    return u.T  # Matriz etiquetas x clusters

def clean_membership_matrix(membership_matrix, threshold=0.1):
    membership_matrix[membership_matrix < threshold] = 0
    row_sums = np.sum(membership_matrix, axis=1, keepdims=True)
    return membership_matrix / np.maximum(row_sums, 1e-6)

## ============================ ORDENAMIENTO DE CLUSTERS ==============================

def order_fuzzy_clusters(cluster_similarities):
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

    return ordered_clusters

## ============================ ENTRENAMIENTO Y PREDICCIÓN =============================

def train_fuzzy_classifiers(X_train, Y_train, membership_matrix, cluster_order, seed=None):
    num_clusters = membership_matrix.shape[1]
    models = []

    scaler = StandardScaler()
    X_train_enhanced = scaler.fit_transform(X_train)

    for cluster_idx in cluster_order:
        print("entrenando cluster ", cluster_idx, " of cluster order: ", cluster_order)
        cluster_weights = membership_matrix[:, cluster_idx]
        selected_labels = cluster_weights > 0

        if np.sum(selected_labels) == 0:
            models.append(None)
            continue

        Y_cluster = Y_train[:, selected_labels]
        model = ClassifierChain(RandomForestClassifier(n_estimators=10, random_state=seed, n_jobs=-1))
        model.fit(X_train_enhanced, Y_cluster)

        Y_pred_cluster = model.predict_proba(X_train_enhanced)
        X_train_enhanced = np.hstack((X_train_enhanced, Y_pred_cluster))

        models.append((model, selected_labels))

    return models, scaler, cluster_order

def predict_fuzzy(models, X_test, membership_matrix, num_labels, scaler, cluster_order):
    num_clusters = len(models)
    X_test_enhanced = scaler.transform(X_test)

    predictions = np.zeros((X_test.shape[0], num_labels))
    weights = np.zeros((X_test.shape[0], num_labels))

    for idx, cluster_idx in enumerate(cluster_order):
        model_entry = models[idx]
        if model_entry is None:
            continue

        model, selected_labels = model_entry
        Y_pred_cluster = model.predict_proba(X_test_enhanced)
        X_test_enhanced = np.hstack((X_test_enhanced, Y_pred_cluster))

        cluster_weights = membership_matrix[:, cluster_idx][selected_labels]
        predictions[:, selected_labels] += Y_pred_cluster * cluster_weights
        weights[:, selected_labels] += cluster_weights

    final_predictions = predictions / np.maximum(weights, 1e-6)
    return (final_predictions > 0.5).astype(int)

## ============================ PIPELINE FCCC ==========================================

def compute_fuzzy_components(Y, num_clusters=3, random_state=None, threshold=0.1):
    """Compute fuzzy clustering components that only need to be calculated once per dataset.
    
    Args:
        Y: Label matrix
        num_clusters: Number of clusters
        random_state: Random seed
        threshold: Membership threshold
        
    Returns:
        tuple: (membership_matrix, cluster_order)
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    similarity_matrix = compute_label_distance_matrix(Y)
    membership_matrix = clean_membership_matrix(apply_fuzzy_cmeans(similarity_matrix, num_clusters, random_state), threshold)
    
    cluster_similarities = compute_fuzzy_cluster_similarity(membership_matrix, similarity_matrix)
    cluster_order = order_fuzzy_clusters(cluster_similarities)
    print("se calcularon los componentes fuzzy")
    
    return membership_matrix, cluster_order

def FCCC(X, Y, X_test, Y_test, num_labels, membership_matrix=None, cluster_order=None, num_clusters=3, random_state=None, threshold=0.1):
    """Fuzzy Clustering Classifier Chain pipeline.
    
    Args:
        X: Training features
        Y: Training labels
        X_test: Test features
        Y_test: Test labels
        num_labels: Number of labels
        membership_matrix: Pre-computed membership matrix (if None, will be computed)
        cluster_order: Pre-computed cluster order (if None, will be computed)
        num_clusters: Number of clusters (only used if membership_matrix is None)
        random_state: Random seed
        threshold: Membership threshold (only used if membership_matrix is None)
        
    Returns:
        tuple: (predictions, evaluation_metrics)
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # Compute fuzzy components if not provided
    if membership_matrix is None or cluster_order is None:
        print("error, esta sección del codigo no debería ejecutarse")
        membership_matrix, cluster_order = compute_fuzzy_components(Y, num_clusters, random_state, threshold)
        
    print("entrenando")
    models, scaler, cluster_order = train_fuzzy_classifiers(X, Y, membership_matrix, cluster_order, random_state)
    print("prediciendo")
    Y_pred_final = predict_fuzzy(models, X_test, membership_matrix, num_labels, scaler, cluster_order)
    print("evaluando")
    evaluation_metrics = evaluate_multilabel_classification(Y_test, Y_pred_final)

    return Y_pred_final, evaluation_metrics
