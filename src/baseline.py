"""Classical Machine Learning Baseline Module.

Implements TF-IDF (word n-grams 1-2) with Logistic Regression, hyperparameter tuning
for regularization parameter C, model saving, and probability estimation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.utils import setup_logger

logger = setup_logger("baseline_model")


def train_baseline_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    c_candidates: Optional[List[float]] = None,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    save_path: str = "models/baseline_model.joblib",
    seed: int = 42
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train TF-IDF + Logistic Regression baseline with validation tuning over C."""
    if c_candidates is None:
        c_candidates = [0.1, 0.5, 1.0, 5.0, 10.0]

    X_train = train_df["text"].values
    y_train = train_df["label"].values.astype(int)
    X_val = val_df["text"].values
    y_val = val_df["label"].values.astype(int)

    logger.info(f"Training TF-IDF vectorizer (max_features={max_features}, ngrams={ngram_range})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    best_c = c_candidates[0]
    best_f1 = -1.0
    best_model = None
    cv_records = []

    logger.info(f"Tuning Logistic Regression regularization C over candidates: {c_candidates}")
    for c in c_candidates:
        clf = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs"
        )
        clf.fit(X_train_vec, y_train)
        val_preds = clf.predict(X_val_vec)
        val_probs = clf.predict_proba(X_val_vec)[:, 1]

        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, val_preds)
        val_auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.5

        logger.info(f"C={c:4.1f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
        cv_records.append({"C": c, "val_acc": val_acc, "val_f1": val_f1, "val_auc": val_auc})

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_c = c
            best_model = clf

    logger.info(f"Best C selected: {best_c} with Validation Macro-F1: {best_f1:.4f}")

    # Build full inference pipeline
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", best_model)
    ])

    # Save artifact
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info(f"Saved baseline pipeline artifact to {save_path}")

    # Final validation metrics
    val_preds = pipeline.predict(X_val)
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    metrics = {
        "best_C": best_c,
        "val_accuracy": float(accuracy_score(y_val, val_preds)),
        "val_precision": float(precision_score(y_val, val_preds, zero_division=0)),
        "val_recall": float(recall_score(y_val, val_preds, zero_division=0)),
        "val_macro_f1": float(f1_score(y_val, val_preds, average="macro", zero_division=0)),
        "val_roc_auc": float(roc_auc_score(y_val, val_probs)),
        "tuning_history": cv_records
    }
    return pipeline, metrics


def load_baseline_model(model_path: str = "models/baseline_model.joblib") -> Pipeline:
    """Load persisted baseline model pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Baseline model not found at {model_path}")
    return joblib.load(model_path)


def predict_baseline(pipeline: Pipeline, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions and positive class (fake) probabilities."""
    clean_texts = [str(t) if str(t).strip() else "empty text" for t in texts]
    probs = pipeline.predict_proba(clean_texts)
    # Binary predictions (0: real, 1: fake)
    preds = pipeline.predict(clean_texts)
    return preds, probs[:, 1]
