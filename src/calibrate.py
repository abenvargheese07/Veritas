"""Model Calibration and Abstention Module.

Implements Expected Calibration Error (ECE), Reliability Diagrams,
and confidence-thresholded abstention ("uncertain" prediction rule)
evaluating coverage vs. accuracy trade-offs.
"""

import os
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.utils import setup_logger

logger = setup_logger("calibration")


def compute_ece(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, List[float], List[float], List[int]]:
    """Compute Expected Calibration Error (ECE) and bin statistics.

    y_probs is probability of class 1. For binary classification, confidence is
    max(P, 1-P) and correctness is whether predicted label matches true label.
    """
    y_preds = (y_probs >= 0.5).astype(int)
    confidences = np.maximum(y_probs, 1.0 - y_probs)
    accuracies = (y_preds == y_true).astype(int)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(float(accuracy_in_bin))
            bin_confs.append(float(avg_confidence_in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)

    return float(ece), bin_accs, bin_confs, bin_counts


def evaluate_abstention(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    abstain_low: float = 0.40,
    abstain_high: float = 0.60
) -> Dict[str, float]:
    """Evaluate abstention rule where model refrains when confidence is near 0.5.

    Abstain when abstain_low <= P(fake) <= abstain_high.
    """
    y_preds = (y_probs >= 0.5).astype(int)
    total_samples = len(y_true)

    # Abstention mask
    abstain_mask = (y_probs >= abstain_low) & (y_probs <= abstain_high)
    covered_mask = ~abstain_mask

    raw_accuracy = float(accuracy_score(y_true, y_preds))
    num_covered = int(np.sum(covered_mask))
    coverage = float(num_covered / total_samples) if total_samples > 0 else 0.0

    if num_covered > 0:
        covered_accuracy = float(accuracy_score(y_true[covered_mask], y_preds[covered_mask]))
    else:
        covered_accuracy = 0.0

    logger.info(
        f"Abstention ({abstain_low}-{abstain_high}): Raw Acc={raw_accuracy:.4f} -> "
        f"Covered Acc={covered_accuracy:.4f} (Coverage={coverage:.2%}, Abstentions={np.sum(abstain_mask)})"
    )

    return {
        "raw_accuracy": raw_accuracy,
        "covered_accuracy": covered_accuracy,
        "accuracy_gain": covered_accuracy - raw_accuracy,
        "coverage": coverage,
        "num_abstained": int(np.sum(abstain_mask)),
        "num_covered": num_covered,
        "total": total_samples
    }


def plot_reliability_diagrams(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str = "results/figures/reliability_diagram.png",
    n_bins: int = 10
) -> Dict[str, float]:
    """Plot reliability curves and ECE comparison for multiple models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_models = len(model_results)
    fig, axes = plt.subplots(1, num_models, figsize=(5.5 * num_models, 4.5), sharey=True)

    if num_models == 1:
        axes = [axes]

    ece_dict = {}
    colors = ["#3498db", "#9b59b6", "#2ecc71"]

    for ax, (model_name, (y_true, y_probs)), color in zip(axes, model_results.items(), colors):
        ece, bin_accs, bin_confs, bin_counts = compute_ece(y_true, y_probs, n_bins=n_bins)
        ece_dict[model_name] = ece

        # Plot ideal diagonal
        ax.plot([0.5, 1.0], [0.5, 1.0], "--", color="gray", label="Perfect Calibration")

        # Plot empirical bin points
        ax.plot(bin_confs, bin_accs, marker="o", linewidth=2, color=color, label=f"{model_name} (ECE={ece:.3f})")
        ax.bar(
            bin_confs,
            bin_accs,
            width=0.04,
            alpha=0.2,
            color=color,
            edgecolor=color
        )

        ax.set_title(f"{model_name}\nECE = {ece:.4f}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Empirical Accuracy")
        ax.set_xlim(0.48, 1.02)
        ax.set_ylim(0.40, 1.02)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved reliability diagram plot to {save_path}")
    return ece_dict
