"""DistilBERT Fine-Tuning and Inference Module.

Fine-tunes Hugging Face distilbert-base-uncased for binary fake news classification
using PyTorch with Apple Silicon MPS / CPU auto-detection, early stopping on validation F1,
learning curve generation, and LIME-compatible inference.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

from src.utils import setup_logger, get_device

logger = setup_logger("distilbert_model")


class PreTokenizedDataset(Dataset):
    """Memory-efficient pre-tokenized tensor dataset."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def train_distilbert_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    epochs: int = 2,
    patience: int = 2,
    subsample_train: Optional[int] = 8000,
    subsample_val: Optional[int] = 1500,
    model_save_dir: str = "models/distilbert_model",
    plot_save_path: str = "results/figures/learning_curves_distilbert.png",
    device: Optional[torch.device] = None,
    seed: int = 42
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, Dict[str, Any]]:
    """Fine-tune DistilBERT on stratified subset with validation tracking and learning curves."""
    if device is None:
        device = get_device()

    logger.info(f"Using device: {device} for DistilBERT fine-tuning ({model_name}).")

    # Stratified subsample to strictly respect hardware & runtime bounds
    def subsample(df: pd.DataFrame, n_samples: Optional[int]) -> pd.DataFrame:
        if n_samples is None or len(df) <= n_samples:
            return df
        n_per_class = n_samples // 2
        parts = []
        for _, group in df.groupby("label"):
            n = min(len(group), n_per_class)
            parts.append(group.sample(n=n, random_state=seed))
        return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_sub = subsample(train_df, subsample_train)
    val_sub = subsample(val_df, subsample_val)
    logger.info(f"DistilBERT dataset sizes: train={len(train_sub)}, val={len(val_sub)}")

    # Tokenizer & Fast Batch Tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Fast pre-tokenizing {len(train_sub)} train samples (max_length={max_length})...")
    train_encodings = tokenizer(
        train_sub["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    val_encodings = tokenizer(
        val_sub["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    train_dataset = PreTokenizedDataset(train_encodings, train_sub["label"].tolist())
    val_dataset = PreTokenizedDataset(val_encodings, val_sub["label"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": []
    }

    best_val_f1 = -1.0
    patience_counter = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        total_train = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)
            total_train += len(labels)

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Epoch {epoch}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        epoch_train_loss = train_loss / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        total_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item() * len(labels)
                total_val += len(labels)

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = accuracy_score(all_labels, all_preds)
        epoch_val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_macro_f1"].append(epoch_val_f1)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | Val Macro-F1: {epoch_val_f1:.4f}"
        )

        # Early stopping on validation macro-F1
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            patience_counter = 0
            best_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered for DistilBERT at epoch {epoch}")
                break

    if best_weights:
        model.load_state_dict(best_weights)

    # Save model and tokenizer
    os.makedirs(model_save_dir, exist_ok=True)
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    logger.info(f"Saved DistilBERT model and tokenizer to {model_save_dir}")

    # Plot learning curves
    if plot_save_path:
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(history["train_loss"], label="Train Loss", marker="o", color="#3498db")
        axes[0].plot(history["val_loss"], label="Val Loss", marker="o", color="#e74c3c")
        axes[0].set_title("DistilBERT Loss Curves", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(history["val_acc"], label="Val Acc", marker="o", color="#2ecc71")
        axes[1].plot(history["val_macro_f1"], label="Val Macro-F1", marker="s", linestyle="--", color="#f39c12")
        axes[1].set_title("DistilBERT Validation Metrics", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(plot_save_path, dpi=200)
        plt.close()
        logger.info(f"Saved DistilBERT learning curves to {plot_save_path}")

    metrics = {
        "best_val_macro_f1": float(best_val_f1),
        "final_val_accuracy": float(history["val_acc"][-1]),
        "epochs_trained": len(history["train_loss"]),
        "history": history
    }
    return model, tokenizer, metrics


def load_distilbert_model(
    model_dir: str = "models/distilbert_model",
    device: Optional[torch.device] = None
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load persisted DistilBERT model and tokenizer."""
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_distilbert(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: Optional[torch.device] = None,
    max_length: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """Inference for DistilBERT returning binary predictions and fake probabilities."""
    if device is None:
        device = get_device()

    model.eval()
    clean_texts = [str(t) if str(t).strip() else "empty text" for t in texts]
    encodings = tokenizer(
        clean_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    dataset = PreTokenizedDataset(encodings, [0] * len(clean_texts))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_preds), np.array(all_probs)
