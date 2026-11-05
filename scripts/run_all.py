#!/usr/bin/env python3
"""End-to-End Fake News Detection Pipeline Runner.

Orchestrates training, experimental hyperparameter sweeps, 4-setting evaluation,
calibration & abstention analysis, failure case extraction, LIME explainability,
and generates figures, tables, and metrics artifacts.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.utils import set_seed, get_device, setup_logger, load_config
from src.data import prepare_and_save_all_data
from src.baseline import train_baseline_model, load_baseline_model, predict_baseline
from src.bilstm import train_bilstm_model, load_bilstm_model, predict_bilstm
from src.bert import train_distilbert_model, load_distilbert_model, predict_distilbert
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrices,
    extract_failure_cases,
    plot_model_comparison
)
from src.calibrate import compute_ece, evaluate_abstention, plot_reliability_diagrams
from src.explain import generate_six_case_studies

logger = setup_logger("run_all")


def run_pipeline(fast_mode: bool = False, config_path: str = "config.yaml") -> Dict[str, Any]:
    """Execute complete end-to-end ML & DL experimentation pipeline."""
    cfg = load_config(config_path)
    seed = cfg["project"]["random_seed"]
    set_seed(seed)
    device = get_device(cfg["hardware"].get("device", "auto"))

    logger.info(f"=== Starting Fake News DL End-Semester Pipeline (Fast Mode={fast_mode}) ===")
    logger.info(f"Compute Device Selected: {device}")

    # Step 1: Ensure Processed Data is Available
    processed_dir = cfg["paths"]["processed_dir"]
    req_files = ["welfake_train.csv", "welfake_val.csv", "welfake_test.csv", "liar_train.csv", "liar_test.csv", "pilot_custom.csv"]
    data_exists = all(os.path.exists(os.path.join(processed_dir, f)) for f in req_files)

    if not data_exists:
        logger.info(f"Processed datasets not found. Preparing data splits...")
        datasets = prepare_and_save_all_data(
            output_dir=processed_dir,
            max_train_samples=cfg["data"]["max_train_samples_per_dataset"],
            max_val_samples=cfg["data"]["max_val_samples_per_dataset"],
            max_test_samples=cfg["data"]["max_test_samples_per_dataset"],
            fast_mode=False,
            seed=seed
        )
        from scripts.download_data import run_eda
        logger.info("Generating EDA figures on full dataset splits...")
        run_eda(datasets, output_dir=cfg["paths"]["figures_dir"])

    # Load data splits
    welfake_train = pd.read_csv(os.path.join(processed_dir, "welfake_train.csv"))
    welfake_val = pd.read_csv(os.path.join(processed_dir, "welfake_val.csv"))
    welfake_test = pd.read_csv(os.path.join(processed_dir, "welfake_test.csv"))

    liar_train = pd.read_csv(os.path.join(processed_dir, "liar_train.csv"))
    liar_val = pd.read_csv(os.path.join(processed_dir, "liar_val.csv"))
    liar_test = pd.read_csv(os.path.join(processed_dir, "liar_test.csv"))

    pilot_custom = pd.read_csv(os.path.join(processed_dir, "pilot_custom.csv"))

    if fast_mode:
        logger.info("Fast mode enabled: subsampling in-memory datasets for rapid sanity verification...")
        welfake_train = welfake_train.head(200)
        welfake_val = welfake_val.head(80)
        welfake_test = welfake_test.head(80)
        liar_train = liar_train.head(200)
        liar_val = liar_val.head(80)
        liar_test = liar_test.head(80)
        pilot_custom = pilot_custom.head(20)

    experiments_log = []
    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["failure_cases_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["metrics_dir"], exist_ok=True)

    # ==========================================
    # STEP 2: MODEL TRAINING & EXPERIMENTS
    # ==========================================

    # --- 1. Classical ML Baseline (TF-IDF + Logistic Regression) ---
    logger.info("--- [Model 1] Training TF-IDF + Logistic Regression Baseline ---")
    base_model_path = os.path.join(models_dir, "baseline_model.joblib")
    if os.path.exists(base_model_path):
        logger.info(f"Loading existing baseline model from {base_model_path}")
        baseline_pipe = load_baseline_model(base_model_path)
        base_preds, base_probs = predict_baseline(baseline_pipe, welfake_val["text"].tolist())
        base_metrics = compute_metrics(welfake_val["label"].values, base_preds, base_probs)
        base_metrics["best_C"] = baseline_pipe.named_steps["clf"].C
        base_metrics["val_accuracy"] = base_metrics["accuracy"]
        base_metrics["val_macro_f1"] = base_metrics["macro_f1"]
        base_metrics["val_roc_auc"] = base_metrics["roc_auc"]
        base_time = 0.0
    else:
        t0 = time.time()
        c_grid = [0.1, 1.0, 5.0] if fast_mode else cfg["models"]["baseline"]["c_candidates"]
        baseline_pipe, base_metrics = train_baseline_model(
            train_df=welfake_train,
            val_df=welfake_val,
            c_candidates=c_grid,
            max_features=cfg["models"]["baseline"]["max_features"],
            save_path=base_model_path,
            seed=seed
        )
        base_time = time.time() - t0
    experiments_log.append({
        "Model_Family": "Baseline_TFIDF_LogReg",
        "Experiment": "Tuned_C_Regularization",
        "Hyperparameters": f"C={base_metrics['best_C']}, ngram=(1,2)",
        "Val_Accuracy": base_metrics["val_accuracy"],
        "Val_Macro_F1": base_metrics["val_macro_f1"],
        "Val_ROC_AUC": base_metrics["val_roc_auc"],
        "Runtime_Seconds": round(base_time, 2)
    })

    # Also train a small baseline on LIAR for cross-dataset Setting B evaluation
    baseline_liar_pipe, _ = train_baseline_model(
        train_df=liar_train,
        val_df=liar_val,
        c_candidates=[1.0],
        save_path=os.path.join(models_dir, "baseline_liar_model.joblib"),
        seed=seed
    )

    # --- 2. Deep Learning: BiLSTM Experiments ---
    logger.info("--- [Model 2] Running BiLSTM Experiments ---")
    bilstm_configs = [
        {"name": "BiLSTM_Base", "hidden_dim": 128, "dropout": 0.3, "augment": False, "epochs": 2 if fast_mode else 8},
        {"name": "BiLSTM_HP_Var1_Hidden64", "hidden_dim": 64, "dropout": 0.3, "augment": False, "epochs": 2 if fast_mode else 5},
        {"name": "BiLSTM_HP_Var2_Drop05", "hidden_dim": 128, "dropout": 0.5, "augment": False, "epochs": 2 if fast_mode else 5},
        {"name": "BiLSTM_Augmentation_WordDropSwap", "hidden_dim": 128, "dropout": 0.3, "augment": True, "epochs": 2 if fast_mode else 5},
    ]

    best_bilstm_f1 = -1.0
    best_bilstm_model = None
    best_bilstm_vocab = None

    for exp in bilstm_configs:
        exp_model_path = os.path.join(models_dir, f"{exp['name']}.pt")
        vocab_path = os.path.join(models_dir, "bilstm_vocab.json")
        t0 = time.time()

        if os.path.exists(exp_model_path) and os.path.exists(vocab_path):
            logger.info(f"Loading existing BiLSTM checkpoint: {exp_model_path}")
            m, v = load_bilstm_model(exp_model_path, vocab_path, device=device)
            val_preds, val_probs = predict_bilstm(m, v, welfake_val["text"].tolist(), device=device)
            val_m = compute_metrics(welfake_val["label"].values, val_preds, val_probs)
            metrics = {
                "final_val_accuracy": val_m["accuracy"],
                "final_val_macro_f1": val_m["macro_f1"]
            }
            exp_time = 0.0
        else:
            logger.info(f"Training BiLSTM Experiment: {exp['name']}")
            m, v, metrics = train_bilstm_model(
                train_df=welfake_train,
                val_df=welfake_val,
                embed_dim=128,
                hidden_dim=exp["hidden_dim"],
                dropout=exp["dropout"],
                learning_rate=cfg["models"]["bilstm"]["learning_rate"],
                batch_size=cfg["models"]["bilstm"]["batch_size"],
                max_length=cfg["models"]["bilstm"]["max_length"],
                epochs=exp["epochs"],
                patience=cfg["models"]["bilstm"]["patience"],
                use_augmentation=exp["augment"],
                model_save_path=exp_model_path,
                vocab_save_path=vocab_path,
                plot_save_path=os.path.join(cfg["paths"]["figures_dir"], f"learning_curves_{exp['name']}.png"),
                device=device,
                seed=seed
            )
            exp_time = time.time() - t0
        experiments_log.append({
            "Model_Family": "BiLSTM",
            "Experiment": exp["name"],
            "Hyperparameters": f"hidden={exp['hidden_dim']}, drop={exp['dropout']}, aug={exp['augment']}",
            "Val_Accuracy": metrics["final_val_accuracy"],
            "Val_Macro_F1": metrics["final_val_macro_f1"],
            "Val_ROC_AUC": np.nan,
            "Runtime_Seconds": round(exp_time, 2)
        })

        if metrics["final_val_macro_f1"] > best_bilstm_f1:
            best_bilstm_f1 = metrics["final_val_macro_f1"]
            best_bilstm_model = m
            best_bilstm_vocab = v
            if not fast_mode:
                # Copy as canonical models/bilstm_model.pt
                torch.save(
                    {
                        "model_state_dict": m.state_dict(),
                        "config": {
                            "vocab_size": len(v),
                            "embed_dim": 128,
                            "hidden_dim": exp["hidden_dim"],
                            "dropout": exp["dropout"],
                            "max_length": cfg["models"]["bilstm"]["max_length"]
                        }
                    },
                    os.path.join(models_dir, "bilstm_model.pt")
                )

    # --- 3. Transformer: DistilBERT Experiments ---
    logger.info("--- [Model 3] Running DistilBERT Fine-Tuning Experiments ---")
    distilbert_configs = [
        {"name": "DistilBERT_Base", "lr": 2e-5, "max_length": 128, "epochs": 1 if fast_mode else 3},
        {"name": "DistilBERT_HP_Var1_LR5e5", "lr": 5e-5, "max_length": 128, "epochs": 1 if fast_mode else 2},
        {"name": "DistilBERT_HP_Var2_Seq64", "lr": 2e-5, "max_length": 64, "epochs": 1 if fast_mode else 2},
    ]

    best_bert_f1 = -1.0
    best_bert_model = None
    best_bert_tokenizer = None

    sub_train = 300 if fast_mode else cfg["models"]["distilbert"]["train_samples"]
    sub_val = 100 if fast_mode else cfg["models"]["distilbert"]["val_samples"]

    for exp in distilbert_configs:
        exp_save_dir = os.path.join(models_dir, exp["name"])
        weight_file = os.path.join(exp_save_dir, "model.safetensors")
        t0 = time.time()

        if os.path.exists(weight_file):
            logger.info(f"Loading existing DistilBERT checkpoint: {exp_save_dir}")
            m, tok = load_distilbert_model(exp_save_dir, device=device)
            val_preds, val_probs = predict_distilbert(m, tok, welfake_val["text"].tolist()[:sub_val], device=device, max_length=exp["max_length"])
            val_labels = welfake_val["label"].values[:sub_val]
            val_m = compute_metrics(val_labels, val_preds, val_probs)
            metrics = {
                "final_val_accuracy": val_m["accuracy"],
                "best_val_macro_f1": val_m["macro_f1"]
            }
            exp_time = 0.0
        else:
            logger.info(f"Training DistilBERT Experiment: {exp['name']}")
            m, tok, metrics = train_distilbert_model(
                train_df=welfake_train,
                val_df=welfake_val,
                model_name=cfg["models"]["distilbert"]["model_name"],
                max_length=exp["max_length"],
                batch_size=cfg["models"]["distilbert"]["batch_size"],
                learning_rate=exp["lr"],
                epochs=exp["epochs"],
                patience=cfg["models"]["distilbert"]["patience"],
                subsample_train=sub_train,
                subsample_val=sub_val,
                model_save_dir=exp_save_dir,
                plot_save_path=os.path.join(cfg["paths"]["figures_dir"], f"learning_curves_{exp['name']}.png"),
                device=device,
                seed=seed
            )
            exp_time = time.time() - t0
        experiments_log.append({
            "Model_Family": "DistilBERT",
            "Experiment": exp["name"],
            "Hyperparameters": f"lr={exp['lr']}, max_len={exp['max_length']}",
            "Val_Accuracy": metrics["final_val_accuracy"],
            "Val_Macro_F1": metrics["best_val_macro_f1"],
            "Val_ROC_AUC": np.nan,
            "Runtime_Seconds": round(exp_time, 2)
        })

        if metrics["best_val_macro_f1"] > best_bert_f1:
            best_bert_f1 = metrics["best_val_macro_f1"]
            best_bert_model = m
            best_bert_tokenizer = tok
            if not fast_mode:
                # Save canonical models/distilbert_model
                canonical_bert_dir = os.path.join(models_dir, "distilbert_model")
                m.save_pretrained(canonical_bert_dir)
                tok.save_pretrained(canonical_bert_dir)

    # Save Experiment Table (only update full experiments table if not fast mode)
    if not fast_mode:
        experiments_df = pd.DataFrame(experiments_log)
        exp_table_path = os.path.join(cfg["paths"]["tables_dir"], "experiments.csv")
        experiments_df.to_csv(exp_table_path, index=False)
        logger.info(f"Saved full experiment table to {exp_table_path}")

    # ==========================================
    # STEP 3: EVALUATION ACROSS FOUR SETTINGS
    # ==========================================
    logger.info("=== Running Comprehensive Evaluation Across 4 Settings ===")

    # Wrapper prediction functions for unified interface
    def get_model_preds(model_name: str, texts: list) -> Tuple[np.ndarray, np.ndarray]:
        if model_name == "Baseline":
            return predict_baseline(baseline_pipe, texts)
        elif model_name == "BiLSTM":
            return predict_bilstm(best_bilstm_model, best_bilstm_vocab, texts, device=device)
        elif model_name == "DistilBERT":
            return predict_distilbert(best_bert_model, best_bert_tokenizer, texts, device=device)
        raise ValueError(f"Unknown model {model_name}")

    models_to_eval = ["Baseline", "BiLSTM", "DistilBERT"]
    comparison_rows = []
    cm_dict = {}

    # --- Setting A: In-Dataset Test Split (WELFake Test) ---
    logger.info("Evaluating Setting A: In-Dataset (WELFake Test)...")
    test_texts_a = welfake_test["text"].tolist()
    y_true_a = welfake_test["label"].values.astype(int)

    setting_a_preds = {}
    for m_name in models_to_eval:
        preds, probs = get_model_preds(m_name, test_texts_a)
        setting_a_preds[m_name] = (preds, probs)
        m = compute_metrics(y_true_a, preds, probs)
        comparison_rows.append({
            "Model": m_name,
            "Setting": "Setting_A_InDataset",
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "Macro_F1": m["macro_f1"],
            "ROC_AUC": m["roc_auc"]
        })
        from sklearn.metrics import confusion_matrix
        cm_dict[f"{m_name} (Setting A)"] = confusion_matrix(y_true_a, preds)

    # --- Setting B: Cross-Dataset Domain Shift ---
    logger.info("Evaluating Setting B: Cross-Dataset (Train WELFake -> Test LIAR)...")
    test_texts_b = liar_test["text"].tolist()
    y_true_b = liar_test["label"].values.astype(int)

    setting_b_preds = {}
    for m_name in models_to_eval:
        preds, probs = get_model_preds(m_name, test_texts_b)
        setting_b_preds[m_name] = (preds, probs)
        m = compute_metrics(y_true_b, preds, probs)
        comparison_rows.append({
            "Model": m_name,
            "Setting": "Setting_B_CrossDataset",
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "Macro_F1": m["macro_f1"],
            "ROC_AUC": m["roc_auc"]
        })

    # Reverse Cross-Dataset Check (Train LIAR -> Test WELFake using baseline)
    rev_preds, rev_probs = predict_baseline(baseline_liar_pipe, test_texts_a)
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

    # --- Setting C: Short vs. Long Text Stratification ---
    logger.info("Evaluating Setting C: Short vs. Long Buckets...")
    short_mask = (welfake_test["length_type"] == "short")
    long_mask = (welfake_test["length_type"] == "long")

    for m_name in models_to_eval:
        preds, probs = setting_a_preds[m_name]
        # Short texts
        if np.sum(short_mask) > 0:
            m_short = compute_metrics(y_true_a[short_mask], preds[short_mask], probs[short_mask])
            comparison_rows.append({
                "Model": m_name,
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
                "Model": m_name,
                "Setting": "Setting_C_LongTexts",
                "Accuracy": m_long["accuracy"],
                "Precision": m_long["precision"],
                "Recall": m_long["recall"],
                "Macro_F1": m_long["macro_f1"],
                "ROC_AUC": m_long["roc_auc"]
            })

    # --- Setting D: Hinglish / WhatsApp Forward Pilot ---
    logger.info("Evaluating Setting D: Custom Hinglish/WhatsApp Pilot Set (N=60)...")
    test_texts_d = pilot_custom["text"].tolist()
    y_true_d = pilot_custom["label"].values.astype(int)

    for m_name in models_to_eval:
        preds, probs = get_model_preds(m_name, test_texts_d)
        m = compute_metrics(y_true_d, preds, probs)
        comparison_rows.append({
            "Model": m_name,
            "Setting": "Setting_D_HinglishPilot",
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "Macro_F1": m["macro_f1"],
            "ROC_AUC": m["roc_auc"]
        })

    # Save Comparison Table & Plot
    comparison_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(cfg["paths"]["tables_dir"], "model_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    logger.info(f"Saved model comparison table to {comp_path}")

    plot_confusion_matrices(cm_dict, save_path=os.path.join(cfg["paths"]["figures_dir"], "confusion_matrices.png"))

    main_settings_df = comparison_df[comparison_df["Setting"].isin([
        "Setting_A_InDataset", "Setting_B_CrossDataset", "Setting_D_HinglishPilot"
    ])]
    plot_model_comparison(main_settings_df, save_path=os.path.join(cfg["paths"]["figures_dir"], "model_comparison.png"))

    # ==========================================
    # STEP 4: CALIBRATION & ABSTENTION ANALYSIS
    # ==========================================
    logger.info("=== Running Calibration & Abstention Analysis ===")
    calib_input = {
        m_name: (y_true_a, setting_a_preds[m_name][1]) for m_name in models_to_eval
    }
    ece_results = plot_reliability_diagrams(
        calib_input, save_path=os.path.join(cfg["paths"]["figures_dir"], "reliability_diagram.png")
    )

    abstain_rows = []
    low = cfg["calibration"]["abstain_low"]
    high = cfg["calibration"]["abstain_high"]

    for m_name in models_to_eval:
        probs = setting_a_preds[m_name][1]
        res = evaluate_abstention(y_true_a, probs, abstain_low=low, abstain_high=high)
        abstain_rows.append({
            "Model": m_name,
            "ECE": ece_results[m_name],
            "Raw_Accuracy": res["raw_accuracy"],
            "Covered_Accuracy": res["covered_accuracy"],
            "Accuracy_Gain": res["accuracy_gain"],
            "Coverage_Percent": res["coverage"] * 100.0,
            "Abstained_Samples": res["num_abstained"],
            "Covered_Samples": res["num_covered"]
        })

    calib_df = pd.DataFrame(abstain_rows)
    calib_path = os.path.join(cfg["paths"]["tables_dir"], "calibration_results.csv")
    calib_df.to_csv(calib_path, index=False)
    logger.info(f"Saved calibration & abstention results to {calib_path}")

    # ==========================================
    # STEP 5: FAILURE ANALYSIS (TOP 30 WRONG)
    # ==========================================
    logger.info("=== Extracting Top 30 Failure Cases Per Model ===")
    for m_name in models_to_eval:
        preds, probs = setting_a_preds[m_name]
        save_p = os.path.join(cfg["paths"]["failure_cases_dir"], f"{m_name.lower()}_failures.csv")
        extract_failure_cases(
            texts=test_texts_a,
            y_true=y_true_a,
            y_pred=preds,
            y_prob=probs,
            model_name=m_name,
            top_n=30,
            save_path=save_p
        )

    # ==========================================
    # STEP 6: EXPLAINABILITY (LIME CASE STUDIES)
    # ==========================================
    logger.info("=== Generating LIME Explainability Studies ===")

    # Function providing class probabilities (N, 2)
    def distilbert_predict_fn(texts: list) -> np.ndarray:
        _, probs_fake = predict_distilbert(best_bert_model, best_bert_tokenizer, texts, device=device)
        return np.vstack([1.0 - probs_fake, probs_fake]).T

    lime_cases = generate_six_case_studies(
        model_name="DistilBERT",
        predict_proba_fn=distilbert_predict_fn,
        test_texts=test_texts_a[:100],  # Sample pool
        y_true=y_true_a[:100],
        output_dir=cfg["paths"]["figures_dir"]
    )
    pd.DataFrame(lime_cases).to_csv(os.path.join(cfg["paths"]["tables_dir"], "lime_cases.csv"), index=False)

    # ==========================================
    # STEP 7: COMPILE ALL METRICS TO JSON
    # ==========================================
    final_metrics_path = os.path.join(cfg["paths"]["metrics_dir"], "final_metrics.json")
    all_summary = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_device": str(device),
        "fast_mode": fast_mode,
        "dataset_sizes": {
            "welfake_train": len(welfake_train),
            "welfake_val": len(welfake_val),
            "welfake_test": len(welfake_test),
            "liar_test": len(liar_test),
            "pilot_custom": len(pilot_custom),
        },
        "model_comparison": comparison_rows,
        "calibration_results": abstain_rows,
        "experiments_summary": experiments_log
    }

    with open(final_metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)
    logger.info(f"Saved master metrics JSON to {final_metrics_path}")

    logger.info("=== Pipeline Completed Successfully! ===")
    return all_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete Fake News detection pipeline.")
    parser.add_argument("--fast", action="store_true", help="Run fast sanity check on smaller subsets.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    run_pipeline(fast_mode=args.fast, config_path=args.config)


if __name__ == "__main__":
    main()
