from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score, roc_curve, auc
import matplotlib.pyplot as plt


def evaluate_binary_classification(y_true, y_pred):
    """Evaluates a binary classifier using various metrics."""
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = precision(tp, tn, fp, fn)
    rec = recall(tp, tn, fp, fn)
    
    return {
        "Accuracy": accuracy(tp, tn, fp, fn),
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1(prec, rec),
        "Specificity": specificity(tp, tn, fp, fn),
        "False Positive Rate (FPR)": false_positive_rate(tn ,fp)
    }

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, tn, fp, fn):
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, tn, fp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def specificity(tp, tn, fp, fn):
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def false_positive_rate(tn, fp):
    return fp / (tn + fp) if (tn + fp) > 0 else 0

def f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
