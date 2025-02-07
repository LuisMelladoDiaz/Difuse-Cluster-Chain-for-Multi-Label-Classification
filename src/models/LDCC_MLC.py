import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from utils.preprocessing import load_arff_data
import skfuzzy as fuzz
from utils.similarity import compute_similarity_matrix

def fuzzy_clustering(similarity_matrix, num_clusters, threshold=0.3):
    """
    Performs fuzzy clustering using scikit-fuzzy (skfuzzy) and assigns labels to clusters based on membership probabilities.
    Labels can belong to multiple clusters if their membership value is above a given threshold.
    """
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(similarity_matrix, num_clusters, 2, error=0.005, maxiter=1000)
    membership_matrix = u.T  # Transpose to align labels with clusters
    
    cluster_assignments = {}
    for label_idx, memberships in enumerate(membership_matrix):
        assigned_clusters = np.where(memberships > threshold)[0]  # Assign labels to clusters where they have significant membership
        cluster_assignments[label_idx] = assigned_clusters.tolist()
    
    return cluster_assignments, membership_matrix

def train_classifiers_per_cluster(X_train, Y_train, cluster_assignments):
    """
    Trains multi-label classifiers for each label cluster using fuzzy memberships.
    """
    models = {}
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train_enhanced = X_train.copy()
    
    for cluster_id, labels in cluster_assignments.items():
        if not labels:
            continue
        
        Y_train_cluster = Y_train[:, labels]
        base_model = LogisticRegression(max_iter=1000)
        cluster_model = MultiOutputClassifier(base_model)
        cluster_model.fit(X_train_enhanced, Y_train_cluster)
        Y_pred_cluster = cluster_model.predict(X_train_enhanced)
        X_train_enhanced = np.hstack((X_train_enhanced, Y_pred_cluster))
        models[cluster_id] = cluster_model
    
    return models

def predict_and_combine(models, X_test, cluster_assignments, membership_matrix, num_labels):
    """
    Performs prediction using trained cluster models and combines results using fuzzy weights.
    Since labels may belong to multiple clusters, predictions are aggregated using a weighted average.
    """
    Y_pred_final = np.zeros((X_test.shape[0], num_labels))
    X_test_enhanced = X_test.copy()
    
    cluster_predictions = {}
    for cluster_id, model in models.items():
        labels = cluster_assignments.get(cluster_id, [])
        if not labels:
            continue
        
        Y_pred_cluster = model.predict(X_test_enhanced)
        X_test_enhanced = np.hstack((X_test_enhanced, Y_pred_cluster))
        
        # Store predictions per label with their membership weight
        for i, label in enumerate(labels):
            if label not in cluster_predictions:
                cluster_predictions[label] = []
            cluster_predictions[label].append((Y_pred_cluster[:, i], membership_matrix[label, cluster_id]))
    
    # Aggregate predictions using membership-based weighted average
    for label, predictions in cluster_predictions.items():
        weighted_sum = sum(pred * weight for pred, weight in predictions)
        total_weight = sum(weight for _, weight in predictions)
        Y_pred_final[:, label] = weighted_sum / total_weight if total_weight > 0 else 0
    
    return Y_pred_final

def LCC_MLC_Fuzzy(file_path, num_labels, sparse=False, num_clusters=10, threshold=0.3):
    """
    Implements the fuzzy Label Cluster Chains for Multi-Label Classification (LCC-MLC) method.
    Steps:
    1. Compute similarity matrix for labels.
    2. Apply fuzzy clustering to group labels into overlapping clusters.
    3. Train classifiers per cluster and propagate predictions.
    4. Combine multiple predictions for labels using membership-based weighting.
    """
    X, Y, features, label_names = load_arff_data(file_path, num_labels, sparse, return_labels_and_features=True)
    similarity_matrix = compute_similarity_matrix(Y)
    cluster_assignments, membership_matrix = fuzzy_clustering(similarity_matrix, num_clusters, threshold)
    models = train_classifiers_per_cluster(X, Y, cluster_assignments)
    Y_pred_final = predict_and_combine(models, X, cluster_assignments, membership_matrix, num_labels)
    
    print("Final fuzzy predictions for all instances:")
    print(Y_pred_final)
    return Y_pred_final