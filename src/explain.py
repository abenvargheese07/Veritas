"""Model Explainability Module Using LIME.

Implements local interpretable model-agnostic explanations (LIME) for text
predictions across Baseline, BiLSTM, and DistilBERT models.
Saves HTML visualizations and extracts word importance scores.
"""

import os
from typing import Dict, List, Tuple, Callable, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

from src.utils import setup_logger

logger = setup_logger("explain")


def get_lime_explainer() -> LimeTextExplainer:
    """Instantiate standard LIME Text Explainer."""
    return LimeTextExplainer(
        class_names=["Real", "Fake"],
        split_expression=r"\W+",
        bow=False,
        random_state=42
    )


def explain_single_prediction(
    text: str,
    predict_proba_fn: Callable[[List[str]], np.ndarray],
    num_features: int = 8,
    num_samples: int = 300
) -> Tuple[Any, List[Tuple[str, float]], float]:
    """Generate LIME explanation for a single text instance."""
    explainer = get_lime_explainer()
    exp = explainer.explain_instance(
        text,
        predict_proba_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=(1,)
    )
    # List of (word, weight) for class 1 (Fake)
    weights = exp.as_list(label=1)
    # Positive weights push toward Fake; negative weights push toward Real
    probs = predict_proba_fn([text])[0]
    fake_prob = float(probs[1])
    return exp, weights, fake_prob


def save_lime_html_and_plot(
    exp: Any,
    text: str,
    true_label: int,
    predicted_label: int,
    model_name: str,
    output_html_path: str,
    output_fig_path: str
) -> None:
    """Save both interactive HTML report and static Matplotlib bar chart."""
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)

    # 1. Save HTML
    exp.save_to_file(output_html_path)

    # 2. Save PNG bar chart of word weights
    weights = exp.as_list(label=1)
    if not weights:
        return

    words = [w for w, _ in weights]
    scores = [s for _, s in weights]

    # Sort by absolute score for display
    idx_sorted = np.argsort(scores)
    sorted_words = [words[i] for i in idx_sorted]
    sorted_scores = [scores[i] for i in idx_sorted]
    colors = ["#2ecc71" if s < 0 else "#e74c3c" for s in sorted_scores]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(sorted_words, sorted_scores, color=colors)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    status_str = "Correct" if true_label == predicted_label else "Wrong"
    ax.set_title(
        f"LIME Word Attributions: {model_name} ({status_str})\n"
        f"True: {true_label} ('{'Fake' if true_label == 1 else 'Real'}') | "
        f"Pred: {predicted_label} ('{'Fake' if predicted_label == 1 else 'Real'}')",
        fontweight="bold",
        fontsize=10
    )
    ax.set_xlabel("Weight (Negative = Real, Positive = Fake)")
    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=200)
    plt.close()


def generate_six_case_studies(
    model_name: str,
    predict_proba_fn: Callable[[List[str]], np.ndarray],
    test_texts: List[str],
    y_true: np.ndarray,
    output_dir: str = "results/figures"
) -> List[Dict[str, Any]]:
    """Generate 6 detailed LIME explanations: 3 correct and 3 incorrect predictions."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating 6 LIME case studies for {model_name}...")

    probs = predict_proba_fn(test_texts)
    preds = np.argmax(probs, axis=1)

    correct_indices = np.where(preds == y_true)[0]
    wrong_indices = np.where(preds != y_true)[0]

    # Select 3 correct (diverse confidences)
    chosen_correct = correct_indices[:3] if len(correct_indices) >= 3 else correct_indices
    # Select 3 wrong (highest confidence errors)
    wrong_confs = [probs[i, preds[i]] for i in wrong_indices]
    chosen_wrong = wrong_indices[np.argsort(wrong_confs)[::-1][:3]] if len(wrong_indices) >= 3 else wrong_indices

    case_studies = []

    case_counter = 1
    for idx_group, is_correct in [(chosen_correct, True), (chosen_wrong, False)]:
        for idx in idx_group:
            text = test_texts[idx]
            t_label = int(y_true[idx])
            p_label = int(preds[idx])

            exp, weights, fake_p = explain_single_prediction(
                text=text,
                predict_proba_fn=predict_proba_fn,
                num_features=6,
                num_samples=200
            )

            tag = "correct" if is_correct else "wrong"
            html_path = os.path.join(output_dir, f"lime_exp_{case_counter}_{tag}_{model_name}.html")
            fig_path = os.path.join(output_dir, f"lime_exp_{case_counter}_{tag}_{model_name}.png")

            save_lime_html_and_plot(exp, text, t_label, p_label, model_name, html_path, fig_path)

            case_studies.append({
                "case_id": case_counter,
                "status": tag,
                "model": model_name,
                "text_snippet": text[:120] + "...",
                "true_label": t_label,
                "predicted_label": p_label,
                "fake_probability": fake_p,
                "top_cues": [w for w, _ in weights[:4]],
                "html_path": html_path,
                "fig_path": fig_path
            })
            case_counter += 1

    logger.info(f"Generated {len(case_studies)} LIME case study artifacts.")
    return case_studies
