"""Evaluation and Failure Analysis Module.

Implements evaluation across the four mandatory experimental settings:
  Setting A: In-dataset test split
  Setting B: Cross-dataset domain shift (quantifies performance drop)
  Setting C: Short vs. Long text bucket stratification
  Setting D: Hinglish and WhatsApp-style forward evaluation

Generates confusion matrices, ROC-AUC curves, failure case extraction,
and model comparison tables/charts.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.utils import setup_logger

logger = setup_logger("evaluate")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, macro-F1, and ROC-AUC."""
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    y_prob = np.array(y_prob).astype(float)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Handle single class edge case in ROC-AUC
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        roc_auc = 0.5

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1_macro,
        "roc_auc": roc_auc
    }


def plot_confusion_matrices(
    cm_dict: Dict[str, np.ndarray],
    save_path: str = "results/figures/confusion_matrices.png"
) -> None:
    """Plot confusion matrices for all evaluated models side-by-side."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_models = len(cm_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(4.5 * num_models, 3.8))

    if num_models == 1:
        axes = [axes]

    labels = ["Real (0)", "Fake (1)"]
    for ax, (model_name, cm) in zip(axes, cm_dict.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax
        )
        ax.set_title(f"{model_name}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrices plot to {save_path}")


def extract_failure_cases(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    top_n: int = 30,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Extract and categorize the top-N most confident incorrect predictions."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    wrong_mask = (y_pred != y_true)
    wrong_indices = np.where(wrong_mask)[0]

    records = []
    for idx in wrong_indices:
        t = texts[idx]
        true_l = int(y_true[idx])
        pred_l = int(y_pred[idx])
        prob = float(y_prob[idx])
        conf = prob if pred_l == 1 else (1.0 - prob)

        # Categorize failure reason based on textual cues
        t_lower = t.lower()
        if any(w in t_lower for w in ["bhai", "forward", "karo", "aaj", "kal", "group", "whatsapp", "UNESCO", "modi ji", "urgent"]):
            cat = "WhatsApp/Hinglish forward pattern mismatch"
        elif len(t.split()) < 20:
            cat = "Short context / brevity ambiguity"
        elif any(w in t_lower for w in ["official", "report", "announced", "minister", "department"]):
            cat = "Authoritative / bureaucratic tone mimicry"
        elif any(w in t_lower for w in ["curse", "miracle", "cure", "cancer", "secret", "video"]):
            cat = "Sensationalist / clickbait lexical cue"
        else:
            cat = "Out-of-domain semantic shift"

        records.append({
            "text": t,
            "true_label": true_l,
            "predicted_label": pred_l,
            "fake_probability": prob,
            "confidence": conf,
            "failure_category": cat,
            "explanation": f"Model predicted {pred_l} with {conf:.2%} confidence despite true label {true_l}."
        })

    df_failures = pd.DataFrame(records)
    if not df_failures.empty:
        df_failures = df_failures.sort_values(by="confidence", ascending=False).head(top_n).reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_failures.to_csv(save_path, index=False)
        logger.info(f"Saved {len(df_failures)} top failure cases for {model_name} to {save_path}")

    return df_failures


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str = "results/figures/model_comparison.png"
) -> None:
    """Create clear bar charts comparing Accuracy and Macro-F1 across models and settings."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Metric 1: Accuracy
    sns.barplot(
        data=comparison_df,
        x="Setting",
        y="Accuracy",
        hue="Model",
        ax=axes[0],
        palette=["#3498db", "#9b59b6", "#2ecc71"]
    )
    axes[0].set_title("Accuracy Across Settings", fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")

    # Metric 2: Macro F1
    sns.barplot(
        data=comparison_df,
        x="Setting",
        y="Macro_F1",
        hue="Model",
        ax=axes[1],
        palette=["#3498db", "#9b59b6", "#2ecc71"]
    )
    axes[1].set_title("Macro-F1 Across Settings", fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved model comparison plot to {save_path}")
