"""Metrics shared by every model. Class 1 = offensive/aggressive, class 0 = not."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),   # headline metric
        "accuracy": accuracy_score(y_true, y_pred),               # reported, but misleading at 82% positives
        "f1_not_offensive": f[0], "f1_offensive": f[1],
        "precision_not_offensive": p[0], "recall_not_offensive": r[0],
        "precision_offensive": p[1], "recall_offensive": r[1],
    }
