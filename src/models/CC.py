from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
import numpy as np
from evaluation_metrics.multilabel_evaluation import evaluate_multilabel_classification
from utils.preprocessing import scale_data
import random

def train_CC(X_train, y_train, X_test, order="random", random_state=None):
    """
    Train a single Classifier Chain (CC) model.

    Args:
        X_train: Training feature matrix.
        y_train: Training label matrix.
        X_test: Test feature matrix.
        order: Order of labels for the chain ("random" or a specific list).
        random_state: Seed for reproducibility.

    Returns:
        Y_pred_chain: Predictions from the classifier chain (2D array).
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    base_lr = LogisticRegression(max_iter=1000, random_state=random_state)
    chain = ClassifierChain(base_lr, order=order, random_state=random_state)
    chain.fit(X_train_scaled, y_train)
    Y_pred_chain = chain.predict(X_test_scaled)
    return Y_pred_chain

def train_ECC(X_train, y_train, X_test, Y_test, number_of_chains=10, random_state=None):
    """
    Train an Ensemble of Classifier Chains (ECC).

    Args:
        X_train: Training feature matrix.
        y_train: Training label matrix.
        X_test: Test feature matrix.
        number_of_chains: Number of individual classifier chains in the ensemble.
        random_state: Seed for reproducibility.

    Returns:
        Y_pred_chains: Predictions from each individual chain (3D array).
        Y_pred_ensemble_binary: Final binary ensemble prediction (2D array).
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    base_lr = LogisticRegression(max_iter=1000, random_state=random_state)
    
    chains = [ClassifierChain(base_lr, order="random", random_state=random_state + i if random_state else None) 
              for i in range(number_of_chains)]
    
    for chain in chains:
        chain.fit(X_train_scaled, y_train)

    Y_pred_chains = np.array([chain.predict(X_test_scaled) for chain in chains])
    Y_pred_ensemble = Y_pred_chains.mean(axis=0)
    Y_pred_ensemble_binary = (Y_pred_ensemble >= 0.5).astype(int)

    evaluation_metrics = evaluate_multilabel_classification(Y_test, Y_pred_ensemble_binary)

    return Y_pred_chains, Y_pred_ensemble_binary, evaluation_metrics
