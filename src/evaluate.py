import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve


def compute_classification_metrics(y_true, y_pred, y_proba):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1_score_value = float((2 * precision * recall) / max(1e-9, precision + recall))
    metrics = {
        "accuracy": float((tp + tn) / max(1, len(y_true))),
        "precision": precision,
        "recall": recall,
        "f1": f1_score_value,
        "classification_report": report,
    }
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    metrics["precision_recall_curve"] = {"precision": precision_curve.tolist(), "recall": recall_curve.tolist()}
    return metrics


def save_confusion_matrix(y_true, y_pred, output_path):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_roc_curve(y_true, y_proba, output_path):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
