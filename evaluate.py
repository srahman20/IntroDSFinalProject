"""
Evaluation utilities for SMS phishing detection.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model_name, y_true, y_pred):
    """
    Calculate standard classification metrics.

    Args:
        model_name: Name of the model.
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing accuracy, precision, recall, and F1-score.
    """
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def print_report(model_name, y_true, y_pred):
    """Print a readable classification report for a model."""
    print(f"\n===== {model_name} Classification Report =====")
    print(classification_report(y_true, y_pred, target_names=["ham", "spam"]))


def save_confusion_matrix(model_name, y_true, y_pred, output_dir="results/figures"):
    """
    Save a confusion matrix visualization for a model.

    Args:
        model_name: Name of the model.
        y_true: True labels.
        y_pred: Predicted labels.
        output_dir: Directory where the figure should be saved.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(Path(output_dir) / f"confusion_matrix_{safe_name}.png", dpi=300)
    plt.close()
