import os
import numpy as np
import skfuzzy as fuzz
import random
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from evaluation_metrics.multilabel_evaluation import evaluate_multilabel_classification
from utils.similarity import compute_label_similarity_matrix

## FUZZY CLUSTERING ###################################################################################################

def apply_fuzzy_cmeans(similarity_matrix, num_clusters, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    max_distance = np.max(similarity_matrix)
    normalized_similarity_matrix = 1 - (similarity_matrix / max_distance)
    
    cntr, u, _, _, _, _, _ = fuzz.cmeans(normalized_similarity_matrix, num_clusters, 1.2, error=0.005, maxiter=1000, seed=seed)
    return u.T  # Matriz etiquetas x clusters


def clean_membership_matrix(membership_matrix, threshold=0.1):
    membership_matrix[membership_matrix < threshold] = 0
    row_sums = np.sum(membership_matrix, axis=1, keepdims=True)
    return membership_matrix / row_sums

## ORDER CLUSTERS ################################################################################################

def order_clusters(num_clusters, seed=None):
    order = list(range(num_clusters))
    if seed is not None:
        random.seed(seed)
    random.shuffle(order)
    return order

## TRAIN & PREDICT ##################################################################

def train_fuzzy_classifiers(X_train, Y_train, membership_matrix, seed=None):
    num_clusters = membership_matrix.shape[1]
    models = []
    cluster_order = order_clusters(num_clusters, seed)
    
    scaler = StandardScaler()
    X_train_enhanced = scaler.fit_transform(X_train)
    
    for cluster_idx in cluster_order:

        # Matriz peso-etiqueta para el cluster x
        cluster_weights = membership_matrix[:, cluster_idx]
        selected_labels = cluster_weights > 0
        
        if np.sum(selected_labels) == 0:
            models.append(None)
            continue
        
        # Entrenar modelo
        Y_cluster = Y_train[:, selected_labels]
        model = ClassifierChain(RandomForestClassifier(n_estimators=100, random_state=seed))
        model.fit(X_train_enhanced, Y_cluster)
        
        # Añadir predicciones como nuevas características
        Y_pred_cluster = model.predict_proba(X_train_enhanced)
        X_train_enhanced = np.hstack((X_train_enhanced, Y_pred_cluster))

        # Añadir modelo a la cadena
        models.append((model, selected_labels))
    
    return models, scaler

def predict_fuzzy(models, X_test, membership_matrix, num_labels, scaler):

    num_clusters = len(models)
    X_test_enhanced = scaler.transform(X_test)

    # Inicializar matriz prediccion
    predictions = np.zeros((X_test.shape[0], num_labels))
    weights = np.zeros((X_test.shape[0], num_labels))
    
    for cluster_idx in range(num_clusters):

        # Seleccionar modelo para el cluster X
        model, selected_labels = models[cluster_idx]

        if model is None:
            continue
        
        # Predecir y añadir como característica
        Y_pred_cluster = model.predict_proba(X_test_enhanced)
        X_test_enhanced = np.hstack((X_test_enhanced, Y_pred_cluster))
        
        # Ponderar predicción
        cluster_weights = membership_matrix[:, cluster_idx][selected_labels]
        predictions[:, selected_labels] += Y_pred_cluster * cluster_weights

        # Accumulación de peso
        weights[:, selected_labels] += cluster_weights

    
    # Normalizar
    final_predictions = predictions / np.maximum(weights, 1e-6) # por si alguna etiqueta no pertenece a ningun cluster despues de aplicar el threshold

    return (final_predictions > 0.5).astype(int)

## FCCC ################################################################################################

def FCCC(X, Y, X_test, Y_test, num_labels, sparse=False, num_clusters=3, random_state=None, threshold=0.1):
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    similarity_matrix = compute_label_similarity_matrix(Y)
    membership_matrix = clean_membership_matrix(apply_fuzzy_cmeans(similarity_matrix, num_clusters, random_state), threshold)
    
    models, scaler = train_fuzzy_classifiers(X, Y, membership_matrix, random_state)
    Y_pred_final = predict_fuzzy(models, X_test, membership_matrix, num_labels, scaler)
    
    evaluation_metrics = evaluate_multilabel_classification(Y_test, Y_pred_final)
    
    return Y_pred_final, evaluation_metrics
