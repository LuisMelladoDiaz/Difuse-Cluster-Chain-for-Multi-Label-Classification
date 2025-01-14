from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import ClassifierChain
import numpy as np
import matplotlib.pyplot as plt

def scale_data(X_train, X_test):
    """
    Escala los datos de entrada usando StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def jaccard_similarity(y_test, y_pred):
    """
    Calcula el índice de similitud de Jaccard.
    """
    return jaccard_score(y_test, y_pred, average="samples", zero_division=0)

def one_vs_rest_classifier(X_train, y_train, X_test):
    """
    Entrena un clasificador OneVsRest y realiza predicciones.
    """
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    base_lr = LogisticRegression(max_iter=1000)
    ovr = OneVsRestClassifier(base_lr)
    ovr.fit(X_train_scaled, y_train)
    y_pred = ovr.predict(X_test_scaled)
    return y_pred

def classifier_chain(X_train, y_train, X_test, y_test, number_of_chains=10):
    """
    Entrena y evalúa un modelo de cadenas de clasificadores.
    """
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    base_lr = LogisticRegression(max_iter=1000)
    chains = [ClassifierChain(base_lr, order="random", random_state=i) for i in range(number_of_chains)]

    for chain in chains:
        chain.fit(X_train_scaled, y_train)

    Y_pred_chains = np.array([chain.predict(X_test_scaled) for chain in chains])

    chain_jaccard_scores = [
        jaccard_similarity(y_test, Y_pred_chain)
        for Y_pred_chain in Y_pred_chains
    ]

    Y_pred_ensemble = Y_pred_chains.mean(axis=0)
    Y_pred_ensemble_binary = (Y_pred_ensemble >= 0.5).astype(int)
    ensemble_jaccard_score = jaccard_similarity(y_test, Y_pred_ensemble_binary)

    print("Jaccard Similarity Scores (Classifier Chains):", chain_jaccard_scores)
    print("Jaccard Similarity Score (Ensemble):", ensemble_jaccard_score)

    return chain_jaccard_scores, ensemble_jaccard_score

def plot_jaccard_scores(ovr_jaccard_score, chain_jaccard_scores, ensemble_jaccard_score, num_chains):
    """
    Plotea la comparación de los índices de similitud de Jaccard.
    """
    model_names = ["Independent"] + [f"Chain {i+1}" for i in range(num_chains)] + ["Ensemble"]
    model_scores = [ovr_jaccard_score] + chain_jaccard_scores + [ensemble_jaccard_score]

    assert len(model_names) == len(model_scores), "El número de nombres y puntuaciones no coincide"

    x_pos = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.grid(True)
    ax.set_title("Classifier Chain Ensemble Performance Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation="vertical")
    ax.set_ylabel("Jaccard Similarity Score")
    ax.set_ylim([min(model_scores) * 0.9, max(model_scores) * 1.1])
    colors = ["r"] + ["b"] * len(chain_jaccard_scores) + ["g"]
    ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
    plt.tight_layout()
    plt.show()
