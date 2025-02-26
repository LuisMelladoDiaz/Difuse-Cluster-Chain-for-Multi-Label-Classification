from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, hamming_loss
)

def evaluate_multilabel_classification(y_true, y_pred):
    """Evaluates a multi-label classifier using common metrics."""
    
    metrics = {
        "Accuracy (Subset)": accuracy_score(y_true, y_pred),  # Exact match (todas las etiquetas correctas)
        "Hamming Loss": hamming_loss(y_true, y_pred),         # Errores a nivel de etiqueta
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-Score (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision (Samples)": precision_score(y_true, y_pred, average="samples", zero_division=0),
        "Recall (Samples)": recall_score(y_true, y_pred, average="samples", zero_division=0),
        "F1-Score (Samples)": f1_score(y_true, y_pred, average="samples", zero_division=0)
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics
