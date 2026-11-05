"""Smoke Tests for Fake News Detection Pipeline.

Verifies data schema integrity, leakage removal, model prediction shapes,
calibration logic, and basic inference functionality.
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

from src.preprocess import clean_leakage_and_noise, assign_length_type
from src.bilstm import BiLSTMClassifier, SimpleVocab
from src.calibrate import compute_ece, evaluate_abstention


def test_leakage_cleaning() -> None:
    """Test that wire service prefixes and reporter signatures are stripped."""
    raw = "WASHINGTON (Reuters) - Federal officials announced new rules today. Reporting by Jane Doe."
    cleaned = clean_leakage_and_noise(raw)
    assert "(reuters)" not in cleaned.lower()
    assert "washington (reuters) -" not in cleaned.lower()
    assert "reporting by" not in cleaned.lower()
    assert "Federal officials announced new rules today." in cleaned


def test_length_type_assignment() -> None:
    """Verify length_type logic: < 50 words is short, >= 50 words is long."""
    short_text = "This is a short news headline."
    long_text = " ".join(["word"] * 60)
    assert assign_length_type(short_text) == "short"
    assert assign_length_type(long_text) == "long"


def test_custom_data_schema() -> None:
    """Check that pilot custom data exists, is balanced, and matches required schema."""
    path = "data/custom/pilot_author_written.csv"
    assert os.path.exists(path), f"{path} must exist"
    df = pd.read_csv(path)
    required_cols = {"text", "label", "source", "length_type"}
    assert required_cols.issubset(set(df.columns))
    assert set(df["label"].unique()) == {0, 1}
    assert len(df) >= 50
    # Balanced
    counts = df["label"].value_counts()
    assert counts[0] == counts[1]


def test_bilstm_architecture_forward() -> None:
    """Verify BiLSTM forward pass and output dimensions."""
    vocab_size = 100
    batch_size = 4
    seq_len = 16
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dim=32,
        num_layers=1,
        dropout=0.2,
        num_classes=2
    )
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(dummy_input)
    assert output.shape == (batch_size, 2)
    assert not torch.isnan(output).any()


def test_calibration_and_ece() -> None:
    """Verify Expected Calibration Error calculation and abstention calculation."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_probs = np.array([0.1, 0.45, 0.85, 0.95, 0.52, 0.2, 0.7, 0.35])

    ece, bin_accs, bin_confs, bin_counts = compute_ece(y_true, y_probs, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert len(bin_accs) == 5

    abstention_results = evaluate_abstention(y_true, y_probs, abstain_low=0.40, abstain_high=0.60)
    assert 0.0 <= abstention_results["coverage"] <= 1.0
    assert abstention_results["num_abstained"] >= 1


def test_processed_data_schema() -> None:
    """Verify processed datasets load with required schema and valid label values."""
    for filename in ["welfake_train.csv", "welfake_test.csv", "liar_test.csv"]:
        path = os.path.join("data", "processed", filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            assert {"text", "label", "source", "length_type"}.issubset(set(df.columns))
            assert set(df["label"].unique()).issubset({0, 1})
            assert not df["text"].isna().any()


def test_baseline_and_bilstm_predictions_shape() -> None:
    """Verify loaded models produce predictions of the expected shape (N,) and valid range."""
    from src.baseline import load_baseline_model, predict_baseline
    from src.bilstm import load_bilstm_model, predict_bilstm

    sample_texts = [
        "Government announces new economic policy today.",
        "URGENT: Forward this secret cure to all WhatsApp groups!",
        "Scientists publish new peer-reviewed findings on solar physics."
    ]
    N = len(sample_texts)

    # 1. Baseline
    baseline_path = "models/baseline_model.joblib"
    if os.path.exists(baseline_path):
        pipe = load_baseline_model(baseline_path)
        preds, probs = predict_baseline(pipe, sample_texts)
        assert len(preds) == N
        assert len(probs) == N
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        assert set(preds).issubset({0, 1})

    # 2. BiLSTM
    bilstm_m_path = "models/bilstm_model.pt"
    bilstm_v_path = "models/bilstm_vocab.json"
    if os.path.exists(bilstm_m_path) and os.path.exists(bilstm_v_path):
        m, v = load_bilstm_model(bilstm_m_path, bilstm_v_path, device="cpu")
        preds, probs = predict_bilstm(m, v, sample_texts, device="cpu")
        assert len(preds) == N
        assert len(probs) == N
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        assert set(preds).issubset({0, 1})


def test_gradio_app_classify_text() -> None:
    """Verify that app/app.py classify_text returns valid tuples for edge cases."""
    from app.app import classify_text

    # 1. Empty input
    verdict, conf, hl, details = classify_text("", "Classical Baseline (TF-IDF + LogReg)")
    assert "Empty Input" in verdict
    assert hl == []

    # 2. Very long input (>5000 characters)
    long_msg = "Breaking news alert: " + "word " * 1200
    verdict, conf, hl, details = classify_text(long_msg, "Classical Baseline (TF-IDF + LogReg)")
    assert isinstance(verdict, str) and len(verdict) > 0
    assert isinstance(conf, str) and "%" in conf
    assert isinstance(details, str)

    # 3. Hinglish message with BiLSTM
    hinglish_msg = "Bhai ye urgent message sabko bhejo, free recharge mil raha hai."
    verdict, conf, hl, details = classify_text(hinglish_msg, "Deep Learning (BiLSTM)")
    assert isinstance(verdict, str) and len(verdict) > 0
    assert isinstance(conf, str) and "%" in conf
    assert isinstance(hl, list)
