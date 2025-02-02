from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(y_true, y_pred):
    """Computes TP, TN, FP, FN from the confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp, tn, fp, fn

def accuracy(tp, tn, fp, fn):
    """Computes accuracy."""
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
    """Computes precision (Positive Predictive Value)."""
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, fn):
    """Computes recall (True Positive Rate or Sensitivity)."""
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(precision, recall):
    """Computes F1-score as the harmonic mean of precision and recall."""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def specificity(tn, fp):
    """Computes specificity (True Negative Rate)."""
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def false_positive_rate(fp, tn):
    """Computes False Positive Rate (FPR)."""
    return fp / (tn + fp) if (tn + fp) > 0 else 0

def evaluate_binary_classification(y_true, y_pred):
    """Evaluates a binary classifier using multiple metrics."""
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)

    acc = accuracy(tp, tn, fp, fn)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(prec, rec)
    spec = specificity(tn, fp)
    fpr = false_positive_rate(fp, tn)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Specificity": spec,
        "False Positive Rate (FPR)": fpr
    }
