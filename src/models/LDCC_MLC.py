import numpy as np
import skfuzzy as fuzz
from utils.preprocessing import load_multilabel_dataset
import random
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from evaluation_metrics.multilabel_evaluation import evaluate_multilabel_classification
from utils.similarity import compute_label_similarity_matrix


## FUZZY CLUSTERING ###################################################################################################

def apply_fuzzy_cmeans(similarity_matrix, num_clusters):

    """Aplica el algoritmo Fuzzy C-Means sobre la matriz de distancias."""
    
    print("Applying Fuzzy c-means over labels...")

    max_distance = np.max(similarity_matrix)
    normalized_similarity_matrix = 1 - (similarity_matrix / max_distance)
    
    cntr, u, _, _, _, _, _ = fuzz.cmeans(normalized_similarity_matrix, num_clusters, 1.2, error=0.005, maxiter=1000)
    membership_matrix = u.T
    
    print(f"Fuzzy clustering completed. Membership matrix shape: {membership_matrix.shape}")
    print(membership_matrix)
    
    return membership_matrix # Matriz etiquetas x clusters


def clean_membership_matrix(membership_matrix, threshold=0.1):
    """Aplica un umbral sobre la matriz de pertenencia y normaliza las filas para que sumen 1."""
    
    membership_matrix[membership_matrix < threshold] = 0
    
    row_sums = np.sum(membership_matrix, axis=1, keepdims=True)
    normalized_membership_matrix = membership_matrix / row_sums
    
    print("Cleaned and normalized membership matrix:")
    print(normalized_membership_matrix)
    
    return normalized_membership_matrix

## ORDER CLUSTERS ################################################################################################
def order_clusters(num_clusters):
    """Devuelve un orden aleatorio de los clusters."""
    order = list(range(num_clusters))
    random.shuffle(order)
    print(f"Cluster order: {order}")
    return order

## LABEL CLUSTER CHAIN ################################################################################################
def train_fuzzy_classifiers(X_train, Y_train, membership_matrix):
    """Entrena un clasificador por cada cluster usando las asignaciones difusas en el orden aleatorio."""
    num_clusters = membership_matrix.shape[1]
    models = []
    cluster_order = order_clusters(num_clusters)
    
    for cluster_idx in cluster_order:
        cluster_weights = membership_matrix[:, cluster_idx]  # Pertenencia de cada etiqueta al cluster
        selected_labels = cluster_weights > 0  # Etiquetas con pertenencia significativa
        
        if np.sum(selected_labels) == 0:
            models.append(None)
            continue
        
        Y_cluster = Y_train[:, selected_labels]  # Subconjunto de etiquetas
        model = ClassifierChain(RandomForestClassifier(n_estimators=100))
        model.fit(X_train, Y_cluster)
        models.append((model, selected_labels))
    
    return models

def predict_fuzzy(models, X_test, membership_matrix, num_labels):
    """Realiza predicciones y combina los resultados ponderando por la pertenencia a cada cluster."""
    num_clusters = len(models)
    predictions = np.zeros((X_test.shape[0], num_labels))
    weights = np.zeros((X_test.shape[0], num_labels))
    
    for cluster_idx in range(num_clusters):
        model, selected_labels = models[cluster_idx]
        if model is None:
            continue
        
        Y_pred = model.predict(X_test)  # Predicciones del cluster ¿PREDICT PROBA??
        cluster_weights = membership_matrix[:, cluster_idx][selected_labels]  # Ponderaciones de pertenencia
        
        predictions[:, selected_labels] += Y_pred * cluster_weights
        weights[:, selected_labels] += cluster_weights
    
    final_predictions = predictions / np.maximum(weights, 1e-6)
    
    return (final_predictions > 0.5).astype(int)  # Binarización final

## Fuzzy_LCC_MLC ################################################################################################

def Fuzzy_LCC_MLC(file_path, num_labels, sparse=False, num_clusters=3):
    """Implementación de Label Cluster Chains con Clustering Difuso."""

    print("Starting Fuzzy LCC-MLC...")

    # Cargar y preprocesar datos
    X, Y, features, label_names = load_multilabel_dataset(file_path, num_labels, sparse)

    # Computar matriz de similitud y aplicar clustering difuso
    similarity_matrix = compute_label_similarity_matrix(Y)
    membership_matrix = clean_membership_matrix(apply_fuzzy_cmeans(similarity_matrix, num_clusters))

    # Entrenar clasificadores
    models = train_fuzzy_classifiers(X, Y, membership_matrix)

    # Realizar predicciones
    Y_pred_final = predict_fuzzy(models, X, membership_matrix, num_labels)

    print("Final predictions for all instances:")
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    print(Y_pred_final)

    evaluate_multilabel_classification(Y, Y_pred_final)

    return Y_pred_final

