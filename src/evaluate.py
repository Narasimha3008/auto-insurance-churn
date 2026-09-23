"""
evaluate.py
-----------
Reusable model evaluation utilities.

Churn is an imbalanced classification problem (~12% positive class).
Accuracy alone is misleading — a model predicting "never churn" hits 88%.
We care about:
  - ROC-AUC: overall discrimination ability
  - Precision-Recall AUC: performance on the minority (churn) class
  - F1 (churn class): balance between catching churners and false alarms
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def print_report(y_true, y_pred, y_prob=None, model_name="Model"):
    """Print a full classification report + AUC scores."""
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Retained", "Churned"]))

    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        print(f"ROC-AUC:           {auc:.4f}")
        print(f"Precision-Recall AUC: {pr_auc:.4f}")


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save=True):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"], ax=ax
    )
    ax.set_title(f"{model_name} — Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save:
        path = FIG_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    plt.show()


def plot_roc_curve(y_true, y_prob, model_name="Model", save=True):
    """Plot ROC curve with AUC score annotated."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        path = FIG_DIR / f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    plt.show()


def plot_precision_recall(y_true, y_prob, model_name="Model", save=True):
    """Plot Precision-Recall curve — more informative than ROC for imbalanced data."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(y=y_true.mean(), color="k", linestyle="--", lw=1, label="Baseline (churn rate)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} — Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    if save:
        path = FIG_DIR / f"{model_name.lower().replace(' ', '_')}_pr_curve.png"
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    plt.show()
