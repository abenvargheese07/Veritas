"""Bidirectional Long Short-Term Memory (BiLSTM) Neural Network Module.

Includes custom vocabulary tokenization, word-level embedding, BiLSTM architecture,
learning rate scheduling, gradient clipping, early stopping, data augmentation,
and learning curve plotting.
"""

import os
import json
import re
import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from src.utils import setup_logger, get_device

logger = setup_logger("bilstm_model")


# --- Tokenizer & Vocabulary ---

class SimpleVocab:
    """Word-level vocabulary builder for BiLSTM."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, max_vocab_size: int = 20000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def tokenize(self, text: str) -> List[str]:
        """Simple regex-based word tokenization."""
        if not isinstance(text, str):
            return []
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, texts: List[str]) -> "SimpleVocab":
        word_counts = {}
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        sorted_words = sorted(
            [w for w, c in word_counts.items() if c >= self.min_freq],
            key=lambda w: word_counts[w],
            reverse=True
        )

        for word in sorted_words[: self.max_vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self

    def text_to_indices(self, text: str, max_length: int = 128) -> List[int]:
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens[:max_length]]
        if len(indices) < max_length:
            indices = indices + [self.word2idx[self.PAD_TOKEN]] * (max_length - len(indices))
        return indices

    def save(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> "SimpleVocab":
        vocab = cls()
        with open(file_path, "r", encoding="utf-8") as f:
            vocab.word2idx = json.load(f)
        vocab.idx2word = {int(idx): word for word, idx in vocab.word2idx.items()}
        return vocab

    def __len__(self) -> int:
        return len(self.word2idx)


# --- Data Augmentation ---

def augment_text(text: str, p_drop: float = 0.1, p_swap: float = 0.1) -> str:
    """Random word dropout and word swap augmentation for text robustness."""
    tokens = re.findall(r"\b\w+\b", str(text))
    if len(tokens) <= 3:
        return text

    # Word dropout
    kept = [t for t in tokens if random.random() > p_drop]
    if len(kept) < 2:
        kept = tokens

    # Word swap
    if len(kept) >= 4 and random.random() < p_swap:
        idx = random.randint(0, len(kept) - 2)
        kept[idx], kept[idx + 1] = kept[idx + 1], kept[idx]

    return " ".join(kept)


# --- PyTorch Dataset & Model ---

class TextSequenceDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: SimpleVocab, max_length: int = 128, augment: bool = False):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        if self.augment:
            text = augment_text(text)
        indices = self.vocab.text_to_indices(text, self.max_length)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with embedding layer and dense head."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2,
        padding_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Global max pooling over sequence length dimension
        pooled, _ = torch.max(lstm_out, dim=1)
        dropped = self.dropout(pooled)
        out = self.relu(self.fc1(dropped))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits


# --- Training Loop ---

def train_bilstm_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    max_length: int = 128,
    epochs: int = 8,
    patience: int = 3,
    grad_clip: float = 5.0,
    use_augmentation: bool = False,
    model_save_path: str = "models/bilstm_model.pt",
    vocab_save_path: str = "models/bilstm_vocab.json",
    plot_save_path: str = "results/figures/learning_curves_bilstm.png",
    device: Optional[torch.device] = None,
    seed: int = 42
) -> Tuple[BiLSTMClassifier, SimpleVocab, Dict[str, Any]]:
    """Train BiLSTM with validation early stopping and learning curves."""
    if device is None:
        device = get_device()

    logger.info(f"Using device: {device} for BiLSTM training.")

    # 1. Build Vocabulary
    vocab = SimpleVocab(max_vocab_size=20000, min_freq=2)
    vocab.fit(train_df["text"].tolist())
    vocab.save(vocab_save_path)
    logger.info(f"Built vocabulary with {len(vocab)} tokens.")

    # 2. Datasets & Loaders
    train_dataset = TextSequenceDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        vocab,
        max_length=max_length,
        augment=use_augmentation
    )
    val_dataset = TextSequenceDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        vocab,
        max_length=max_length,
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model setup
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=dropout,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": []
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    logger.info(
        f"Training BiLSTM: epochs={epochs}, hidden_dim={hidden_dim}, dropout={dropout}, "
        f"lr={learning_rate}, augmentation={use_augmentation}"
    )

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            total_train += inputs.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()
                total_val += inputs.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val
        scheduler.step(epoch_val_loss)

        # Compute macro F1
        from sklearn.metrics import f1_score
        epoch_val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        history["val_macro_f1"].append(epoch_val_f1)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f} | Val F1: {epoch_val_f1:.4f}"
        )

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Restore best weights
    if best_weights:
        model.load_state_dict(best_weights)

    # Save model checkpoint
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": len(vocab),
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "max_length": max_length,
            },
        },
        model_save_path,
    )
    logger.info(f"Saved BiLSTM checkpoint to {model_save_path}")

    # Plot learning curves
    if plot_save_path:
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(history["train_loss"], label="Train Loss", marker="o", color="#3498db")
        axes[0].plot(history["val_loss"], label="Val Loss", marker="o", color="#e74c3c")
        axes[0].set_title("BiLSTM Loss Curves", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(history["train_acc"], label="Train Acc", marker="o", color="#3498db")
        axes[1].plot(history["val_acc"], label="Val Acc", marker="o", color="#2ecc71")
        axes[1].plot(history["val_macro_f1"], label="Val Macro-F1", marker="s", linestyle="--", color="#f39c12")
        axes[1].set_title("BiLSTM Accuracy & F1 Curves", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(plot_save_path, dpi=200)
        plt.close()
        logger.info(f"Saved BiLSTM learning curve plot to {plot_save_path}")

    metrics = {
        "best_val_loss": float(best_val_loss),
        "final_val_accuracy": float(history["val_acc"][-1]),
        "final_val_macro_f1": float(history["val_macro_f1"][-1]),
        "epochs_trained": len(history["train_loss"]),
        "history": history
    }
    return model, vocab, metrics


def load_bilstm_model(
    model_path: str = "models/bilstm_model.pt",
    vocab_path: str = "models/bilstm_vocab.json",
    device: Optional[torch.device] = None
) -> Tuple[BiLSTMClassifier, SimpleVocab]:
    """Load trained BiLSTM model and vocabulary."""
    if device is None:
        device = get_device()

    vocab = SimpleVocab.load(vocab_path)
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint["config"]

    model = BiLSTMClassifier(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        num_classes=2
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


def predict_bilstm(
    model: BiLSTMClassifier,
    vocab: SimpleVocab,
    texts: List[str],
    device: Optional[torch.device] = None,
    max_length: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """Inference for BiLSTM returning binary predictions and fake probabilities."""
    if device is None:
        device = get_device()

    model.eval()
    dataset = TextSequenceDataset(texts, [0] * len(texts), vocab, max_length=max_length)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            # Probability of Fake (class 1)
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_preds), np.array(all_probs)
