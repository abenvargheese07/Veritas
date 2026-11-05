#!/usr/bin/env python3
"""Comprehensive Multi-Setting Evaluation, Calibration, and Explainability Runner.

Evaluates Baseline, BiLSTM, and DistilBERT across Settings A, B, C, and D,
computes ECE, generates reliability diagrams and abstention metrics,
extracts top-30 failure cases, runs LIME explanations, and outputs
comparison tables and charts.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.utils import set_seed, get_device, setup_logger, load_config
from src.baseline import load_baseline_model, predict_baseline
from src.bilstm import load_bilstm_model, predict_bilstm
from src.bert import load_distilbert_model, predict_distilbert
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrices,
    extract_failure_cases,
    plot_model_comparison
)
from src.calibrate import compute_ece, evaluate_abstention, plot_reliability_diagrams
from src.explain import generate_six_case_studies

logger = setup_logger("evaluate_all")


def main():
    cfg = load_config("config.yaml")
    seed = cfg["project"]["random_seed"]
    set_seed(seed)
    device = get_device(cfg["hardware"].get("device", "auto"))
    logger.info(f"Evaluation Runner started on Device: {device}")

    models_dir = cfg["paths"]["models_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    tables_dir = cfg["paths"]["tables_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    failure_cases_dir = cfg["paths"]["failure_cases_dir"]

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(failure_cases_dir, exist_ok=True)

    # 1. Load Datasets
    logger.info("Loading processed evaluation datasets...")
    welfake_test = pd.read_csv(os.path.join(processed_dir, "welfake_test.csv"))
    liar_test = pd.read_csv(os.path.join(processed_dir, "liar_test.csv"))
    pilot_custom = pd.read_csv(os.path.join(processed_dir, "pilot_custom.csv"))

    logger.info(
        f"Datasets loaded: WELFake Test={len(welfake_test)}, LIAR Test={len(liar_test)}, "
        f"Pilot Custom={len(pilot_custom)}"
    )

    # 2. Load Models
    logger.info("Loading trained models from disk...")
    baseline_pipe = load_baseline_model(os.path.join(models_dir, "baseline_model.joblib"))
    baseline_liar_pipe = load_baseline_model(os.path.join(models_dir, "baseline_liar_model.joblib"))
    bilstm_model, bilstm_vocab = load_bilstm_model(
        os.path.join(models_dir, "bilstm_model.pt"),
        os.path.join(models_dir, "bilstm_vocab.json"),
        device=device
    )
    distilbert_model, distilbert_tok = load_distilbert_model(
        os.path.join(models_dir, "distilbert_model"),
        device=device
    )
    logger.info("All models loaded successfully!")

    # Helper prediction wrapper
    def get_preds(m_name: str, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if m_name == "Baseline":
            return predict_baseline(baseline_pipe, texts)
        elif m_name == "BiLSTM":
            return predict_bilstm(bilstm_model, bilstm_vocab, texts, device=device)
        elif m_name == "DistilBERT":
            return predict_distilbert(distilbert_model, distilbert_tok, texts, device=device)
        raise ValueError(f"Unknown model {m_name}")

    models = ["Baseline", "BiLSTM", "DistilBERT"]
    comparison_rows = []
    cm_dict = {}

    # ==========================================
    # SETTING A: In-Dataset (WELFake Test)
    # ==========================================
    logger.info("--- Evaluating Setting A: In-Dataset (WELFake Test) ---")
    texts_a = welfake_test["text"].tolist()
    y_true_a = welfake_test["label"].values.astype(int)

    setting_a_results = {}
    for m in models:
        preds, probs = get_preds(m, texts_a)
        setting_a_results[m] = (preds, probs)
        metrics = compute_metrics(y_true_a, preds, probs)
        comparison_rows.append({
            "Model": m,
            "Setting": "Setting_A_InDataset",
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Macro_F1": metrics["macro_f1"],
            "ROC_AUC": metrics["roc_auc"]
        })
        from sklearn.metrics import confusion_matrix
        cm_dict[f"{m} (Setting A)"] = confusion_matrix(y_true_a, preds)
        logger.info(f"Setting A [{m}]: Acc={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}, AUC={metrics['roc_auc']:.4f}")

    # ==========================================
    # SETTING B: Cross-Dataset Domain Shift
    # ==========================================
    logger.info("--- Evaluating Setting B: Cross-Dataset (Train WELFake -> Test LIAR) ---")
    texts_b = liar_test["text"].tolist()
    y_true_b = liar_test["label"].values.astype(int)

    setting_b_results = {}
    for m in models:
        preds, probs = get_preds(m, texts_b)
        setting_b_results[m] = (preds, probs)
        metrics = compute_metrics(y_true_b, preds, probs)
        comparison_rows.append({
            "Model": m,
            "Setting": "Setting_B_CrossDataset",
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Macro_F1": metrics["macro_f1"],
            "ROC_AUC": metrics["roc_auc"]
        })
        logger.info(f"Setting B [{m}]: Acc={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}, AUC={metrics['roc_auc']:.4f}")

    # Reverse Cross-Dataset Check (Train LIAR -> Test WELFake)
    rev_preds, rev_probs = predict_baseline(baseline_liar_pipe, texts_a)
    rev_m = compute_metrics(y_true_a, rev_preds, rev_probs)
    comparison_rows.append({
        "Model": "Baseline_TrainedOnLIAR",
        "Setting": "Setting_B_ReverseCross",
        "Accuracy": rev_m["accuracy"],
        "Precision": rev_m["precision"],
        "Recall": rev_m["recall"],
        "Macro_F1": rev_m["macro_f1"],
        "ROC_AUC": rev_m["roc_auc"]
    })
    logger.info(f"Setting B [Reverse Baseline]: Acc={rev_m['accuracy']:.4f}, Macro-F1={rev_m['macro_f1']:.4f}")

    # ==========================================
    # SETTING C: Short vs. Long Text Stratification
    # ==========================================
    logger.info("--- Evaluating Setting C: Short vs. Long Buckets ---")
    short_mask = (welfake_test["length_type"] == "short").values
    long_mask = (welfake_test["length_type"] == "long").values

    for m in models:
        preds, probs = setting_a_results[m]
        # Short texts
        if np.sum(short_mask) > 0:
            m_short = compute_metrics(y_true_a[short_mask], preds[short_mask], probs[short_mask])
            comparison_rows.append({
                "Model": m,
                "Setting": "Setting_C_ShortTexts",
                "Accuracy": m_short["accuracy"],
                "Precision": m_short["precision"],
                "Recall": m_short["recall"],
                "Macro_F1": m_short["macro_f1"],
                "ROC_AUC": m_short["roc_auc"]
            })
        # Long texts
        if np.sum(long_mask) > 0:
            m_long = compute_metrics(y_true_a[long_mask], preds[long_mask], probs[long_mask])
            comparison_rows.append({
                "Model": m,
                "Setting": "Setting_C_LongTexts",
                "Accuracy": m_long["accuracy"],
                "Precision": m_long["precision"],
                "Recall": m_long["recall"],
                "Macro_F1": m_long["macro_f1"],
                "ROC_AUC": m_long["roc_auc"]
            })

    # ==========================================
    # SETTING D: Hinglish / WhatsApp Forward Pilot (N=60)
    # ==========================================
    logger.info("--- Evaluating Setting D: Custom Hinglish/WhatsApp Pilot Set (N=60) ---")
    texts_d = pilot_custom["text"].tolist()
    y_true_d = pilot_custom["label"].values.astype(int)

    for m in models:
        preds, probs = get_preds(m, texts_d)
        m_pilot = compute_metrics(y_true_d, preds, probs)
        comparison_rows.append({
            "Model": m,
            "Setting": "Setting_D_HinglishPilot",
            "Accuracy": m_pilot["accuracy"],
            "Precision": m_pilot["precision"],
            "Recall": m_pilot["recall"],
            "Macro_F1": m_pilot["macro_f1"],
            "ROC_AUC": m_pilot["roc_auc"]
        })
        logger.info(f"Setting D [{m}]: Acc={m_pilot['accuracy']:.4f}, Macro-F1={m_pilot['macro_f1']:.4f}")

    # Save Comparison Table & Plots
    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(tables_dir, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Saved model comparison table to {comp_path}")

    plot_confusion_matrices(cm_dict, save_path=os.path.join(figures_dir, "confusion_matrices.png"))

    main_settings_df = comp_df[comp_df["Setting"].isin([
        "Setting_A_InDataset", "Setting_B_CrossDataset", "Setting_D_HinglishPilot"
    ])]
    plot_model_comparison(main_settings_df, save_path=os.path.join(figures_dir, "model_comparison.png"))

    # ==========================================
    # CALIBRATION & ABSTENTION ANALYSIS
    # ==========================================
    logger.info("--- Running Calibration & Abstention Analysis ---")
    calib_input = {m: (y_true_a, setting_a_results[m][1]) for m in models}
    ece_results = plot_reliability_diagrams(
        calib_input, save_path=os.path.join(figures_dir, "reliability_diagram.png")
    )

    abstain_rows = []
    low = cfg["calibration"].get("abstain_low", 0.40)
    high = cfg["calibration"].get("abstain_high", 0.60)

    for m in models:
        probs = setting_a_results[m][1]
        res = evaluate_abstention(y_true_a, probs, abstain_low=low, abstain_high=high)
        abstain_rows.append({
            "Model": m,
            "ECE": ece_results[m],
            "Raw_Accuracy": res["raw_accuracy"],
            "Covered_Accuracy": res["covered_accuracy"],
            "Accuracy_Gain": res["accuracy_gain"],
            "Coverage_Percent": res["coverage"] * 100.0,
            "Abstained_Samples": res["num_abstained"],
            "Covered_Samples": res["num_covered"]
        })

    calib_df = pd.DataFrame(abstain_rows)
    calib_path = os.path.join(tables_dir, "calibration_results.csv")
    calib_df.to_csv(calib_path, index=False)
    logger.info(f"Saved calibration & abstention results to {calib_path}")

    # ==========================================
    # FAILURE CASE EXTRACTION (TOP 30 WRONG)
    # ==========================================
    logger.info("--- Extracting Top 30 Failure Cases Per Model ---")
    for m in models:
        preds, probs = setting_a_results[m]
        extract_failure_cases(
            texts=texts_a,
            y_true=y_true_a,
            y_pred=preds,
            y_prob=probs,
            model_name=m,
            top_n=30,
            save_path=os.path.join(failure_cases_dir, f"{m.lower()}_failures.csv")
        )

    # ==========================================
    # EXPLAINABILITY (LIME CASE STUDIES)
    # ==========================================
    logger.info("--- Generating LIME Case Studies ---")
    def distilbert_predict_fn(t_list: list) -> np.ndarray:
        _, p_fake = predict_distilbert(distilbert_model, distilbert_tok, t_list, device=device)
        return np.vstack([1.0 - p_fake, p_fake]).T

    lime_cases = generate_six_case_studies(
        model_name="DistilBERT",
        predict_proba_fn=distilbert_predict_fn,
        test_texts=texts_a[:100],
        y_true=y_true_a[:100],
        output_dir=figures_dir
    )
    pd.DataFrame(lime_cases).to_csv(os.path.join(tables_dir, "lime_cases.csv"), index=False)

    # ==========================================
    # SAVE CONSOLIDATED METRICS JSON
    # ==========================================
    final_metrics = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_device": str(device),
        "dataset_sizes": {
            "welfake_test": len(welfake_test),
            "liar_test": len(liar_test),
            "pilot_custom": len(pilot_custom),
        },
        "model_comparison": comparison_rows,
        "calibration_results": abstain_rows,
        "ece_scores": ece_results
    }

    with open(os.path.join(metrics_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("=== Comprehensive Evaluation Completed Successfully! ===")


if __name__ == "__main__":
    main()
