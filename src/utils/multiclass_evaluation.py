from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def compute_confusion_matrix_multiclass(y_true, y_pred):
    """Computes the confusion matrix for multi-class classification."""
    return confusion_matrix(y_true, y_pred)

def accuracy_multiclass(y_true, y_pred):
    """Computes accuracy for multi-class classification."""
    return accuracy_score(y_true, y_pred)

def precision_multiclass(y_true, y_pred):
    """Computes macro-averaged precision for multi-class classification."""
    return precision_score(y_true, y_pred, average="macro")

def recall_multiclass(y_true, y_pred):
    """Computes macro-averaged recall for multi-class classification."""
    return recall_score(y_true, y_pred, average="macro")

def f1_score_multiclass(y_true, y_pred):
    """Computes macro-averaged F1-score for multi-class classification."""
    return f1_score(y_true, y_pred, average="macro")

def cohen_kappa(y_true, y_pred):
    """Computes Cohen's Kappa coefficient to measure agreement."""
    return cohen_kappa_score(y_true, y_pred)

def evaluate_multiclass_classification(y_true, y_pred):
    """Evaluates a multi-class classifier using multiple metrics."""
    conf_matrix = compute_confusion_matrix_multiclass(y_true, y_pred)

    acc = accuracy_multiclass(y_true, y_pred)
    prec = precision_multiclass(y_true, y_pred)
    rec = recall_multiclass(y_true, y_pred)
    f1 = f1_score_multiclass(y_true, y_pred)
    kappa = cohen_kappa(y_true, y_pred)

    return {
        "Confusion Matrix": conf_matrix,
        "Accuracy": acc,
        "Precision (Macro)": prec,
        "Recall (Macro)": rec,
        "F1-Score (Macro)": f1,
        "Cohen's Kappa": kappa
    }
