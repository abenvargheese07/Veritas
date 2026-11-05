#!/usr/bin/env python3
"""Dedicated BiLSTM Training Script.

Trains base BiLSTM, hyperparameter variations (hidden size, dropout),
and word dropout/swap data augmentation run on the full processed dataset.
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
from src.bilstm import train_bilstm_model, load_bilstm_model, predict_bilstm
from src.evaluate import compute_metrics

logger = setup_logger("train_bilstm")


def main():
    cfg = load_config("config.yaml")
    seed = cfg["project"]["random_seed"]
    set_seed(seed)
    device = get_device(cfg["hardware"].get("device", "auto"))
    logger.info(f"BiLSTM Training Runner on Device: {device}")

    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    welfake_train = pd.read_csv(os.path.join(processed_dir, "welfake_train.csv"))
    welfake_val = pd.read_csv(os.path.join(processed_dir, "welfake_val.csv"))
    logger.info(f"Loaded training data: {len(welfake_train)} rows, val: {len(welfake_val)} rows")

    bilstm_configs = [
        {"name": "BiLSTM_Base", "hidden_dim": 128, "dropout": 0.3, "augment": False, "epochs": 6},
        {"name": "BiLSTM_HP_Var1_Hidden64", "hidden_dim": 64, "dropout": 0.3, "augment": False, "epochs": 5},
        {"name": "BiLSTM_HP_Var2_Drop05", "hidden_dim": 128, "dropout": 0.5, "augment": False, "epochs": 5},
        {"name": "BiLSTM_Augmentation_WordDropSwap", "hidden_dim": 128, "dropout": 0.3, "augment": True, "epochs": 5},
    ]

    vocab_path = os.path.join(models_dir, "bilstm_vocab.json")
    best_f1 = -1.0
    best_checkpoint = None

    for exp in bilstm_configs:
        exp_path = os.path.join(models_dir, f"{exp['name']}.pt")
        # Check if already trained on full dataset
        if os.path.exists(exp_path) and os.path.exists(vocab_path):
            try:
                m, v = load_bilstm_model(exp_path, vocab_path, device=device)
                preds, probs = predict_bilstm(m, v, welfake_val["text"].tolist(), device=device)
                m_metrics = compute_metrics(welfake_val["label"].values, preds, probs)
                logger.info(f"Existing checkpoint {exp['name']}: Val Acc={m_metrics['accuracy']:.4f}, Macro-F1={m_metrics['macro_f1']:.4f}")
                if m_metrics["macro_f1"] > best_f1:
                    best_f1 = m_metrics["macro_f1"]
                    best_checkpoint = exp_path
                continue
            except Exception as e:
                logger.warning(f"Failed to load {exp_path}, re-training: {e}")

        logger.info(f"--- Training {exp['name']} ---")
        t0 = time.time()
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
            model_save_path=exp_path,
            vocab_save_path=vocab_path,
            plot_save_path=os.path.join(cfg["paths"]["figures_dir"], f"learning_curves_{exp['name']}.png"),
            device=device,
            seed=seed
        )
        elapsed = time.time() - t0
        logger.info(f"Completed {exp['name']} in {elapsed:.1f}s | Val F1: {metrics['final_val_macro_f1']:.4f}")

        if metrics["final_val_macro_f1"] > best_f1:
            best_f1 = metrics["final_val_macro_f1"]
            best_checkpoint = exp_path

    # Ensure canonical models/bilstm_model.pt is pointing to the best model
    if best_checkpoint:
        logger.info(f"Setting canonical BiLSTM model from {best_checkpoint} (F1={best_f1:.4f})")
        ckpt = torch.load(best_checkpoint, map_location="cpu")
        torch.save(ckpt, os.path.join(models_dir, "bilstm_model.pt"))

    logger.info("All BiLSTM models trained and saved successfully!")


if __name__ == "__main__":
    main()
