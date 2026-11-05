#!/usr/bin/env python3
"""Dedicated DistilBERT Fine-Tuning Script.

Fine-tunes DistilBERT base uncased and hyperparameter variations (learning rate, sequence length)
on the stratified subset (8,000 samples) using Apple Silicon MPS / CPU auto-detection.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.utils import set_seed, get_device, setup_logger, load_config
from src.bert import train_distilbert_model, load_distilbert_model, predict_distilbert
from src.evaluate import compute_metrics

logger = setup_logger("train_bert")


def main():
    cfg = load_config("config.yaml")
    seed = cfg["project"]["random_seed"]
    set_seed(seed)
    device = get_device(cfg["hardware"].get("device", "auto"))
    logger.info(f"DistilBERT Training Runner on Device: {device}")

    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    welfake_train = pd.read_csv(os.path.join(processed_dir, "welfake_train.csv"))
    welfake_val = pd.read_csv(os.path.join(processed_dir, "welfake_val.csv"))
    logger.info(f"Loaded training data: {len(welfake_train)} rows, val: {len(welfake_val)} rows")

    sub_train = cfg["models"]["distilbert"].get("train_samples", 8000)
    sub_val = cfg["models"]["distilbert"].get("val_samples", 1500)

    distilbert_configs = [
        {"name": "DistilBERT_Base", "lr": 2e-5, "max_length": 128, "epochs": 2},
        {"name": "DistilBERT_HP_Var1_LR5e5", "lr": 5e-5, "max_length": 128, "epochs": 2},
        {"name": "DistilBERT_HP_Var2_Seq64", "lr": 2e-5, "max_length": 64, "epochs": 2},
    ]

    best_f1 = -1.0
    best_config_dir = None

    for exp in distilbert_configs:
        exp_save_dir = os.path.join(models_dir, exp["name"])
        weight_file = os.path.join(exp_save_dir, "model.safetensors")

        if os.path.exists(weight_file):
            try:
                m, tok = load_distilbert_model(exp_save_dir, device=device)
                preds, probs = predict_distilbert(m, tok, welfake_val["text"].tolist()[:sub_val], device=device, max_length=exp["max_length"])
                val_m = compute_metrics(welfake_val["label"].values[:sub_val], preds, probs)
                logger.info(f"Existing checkpoint {exp['name']}: Val Acc={val_m['accuracy']:.4f}, Macro-F1={val_m['macro_f1']:.4f}")
                if val_m["macro_f1"] > best_f1:
                    best_f1 = val_m["macro_f1"]
                    best_config_dir = exp_save_dir
                continue
            except Exception as e:
                logger.warning(f"Failed to load {exp_save_dir}, re-training: {e}")

        logger.info(f"--- Training {exp['name']} (lr={exp['lr']}, max_len={exp['max_length']}, epochs={exp['epochs']}) ---")
        t0 = time.time()
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
        elapsed = time.time() - t0
        logger.info(f"Completed {exp['name']} in {elapsed:.1f}s | Best Val F1: {metrics['best_val_macro_f1']:.4f}")

        if metrics["best_val_macro_f1"] > best_f1:
            best_f1 = metrics["best_val_macro_f1"]
            best_config_dir = exp_save_dir

    # Set canonical model
    if best_config_dir:
        canonical_dir = os.path.join(models_dir, "distilbert_model")
        logger.info(f"Setting canonical DistilBERT model from {best_config_dir} (F1={best_f1:.4f})")
        m, tok = load_distilbert_model(best_config_dir, device="cpu")
        m.save_pretrained(canonical_dir)
        tok.save_pretrained(canonical_dir)

    logger.info("All DistilBERT models fine-tuned and saved successfully!")


if __name__ == "__main__":
    main()
