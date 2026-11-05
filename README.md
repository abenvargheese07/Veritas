# Fake News Detection Using Deep Learning: A Comparative Study of Classical ML, BiLSTM, and DistilBERT on Real-World, Hinglish-Aware News and Forwards

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Deep Learning end-semester project for **B.Sc. Data Science & AI (5th Semester, Artificial Neural Networks & Deep Learning)**.

---

## 🎯 Executive Overview & Core Problem

Recent machine learning models often report ~99% accuracy on popular fake news benchmarks. However, this high score is frequently an empirical mirage caused by **publisher wire leakage** (e.g., `(Reuters)` stamps), narrow single-source domains, and stylistic artifacts. When tested on out-of-domain political claims, short headlines, or informal multilingual (Hinglish) WhatsApp forwards, performance drops drastically.

This project delivers:
1. **Three Model Families**: Classical ML Baseline (TF-IDF + Logistic Regression), Deep Recurrent DL (BiLSTM with word embeddings and data augmentation), and Pretrained Transformer (`distilbert-base-uncased` fine-tuned on Apple Silicon MPS / CPU).
2. **Aggressive Leakage Elimination**: Automated regex cleaning of wire prefixes, reporter bylines, URLs, and location stamps.
3. **Four-Setting Evaluation**:
   - **Setting A**: In-dataset test split (WELFake: 2,000 articles)
   - **Setting B**: Cross-dataset domain shift (WELFake $\leftrightarrow$ LIAR: 2,296 claims)
   - **Setting C**: Document length stratification ($<50$ words vs. $\ge 50$ words)
   - **Setting D**: Author-written pilot benchmark of Hinglish and WhatsApp-style forwards ($N=60$)
4. **Uncertainty Calibration & Abstention**: Expected Calibration Error (ECE) measurement and a $[0.40, 0.60]$ confidence band abstention mechanism (*"Not sure / Inconclusive"*).
5. **Explainability**: Local Interpretable Model-agnostic Explanations (LIME) analyzing top positive and negative word attributions across correct and incorrect predictions.
6. **Production Gradio App**: Interactive web app with model switcher, dynamic word highlighting, confidence scores, and mandatory fact-checking screening disclaimers.

---

## 📊 Summary of Key Results

All numbers below are strictly empirical, reproducible, and recorded in `results/tables/model_comparison.csv` and `results/metrics/final_metrics.json`.

| Evaluation Setting | Classical Baseline (TF-IDF + LogReg) | Deep Learning (BiLSTM) | Transformer (DistilBERT) |
| :--- | :---: | :---: | :---: |
| **Setting A: In-Dataset (WELFake)** | 93.55% Acc / 0.9355 F1 | 93.80% Acc / 0.9380 F1 | **98.80% Acc / 0.9880 F1** |
| **Setting B: Cross-Dataset (LIAR Shift)** | 57.32% Acc / 0.4894 F1 | 57.71% Acc / 0.5185 F1 | **56.10% Acc / 0.5595 F1** |
| *Macro-F1 Drop from Setting A to B* | *-44.61 points* | *-41.95 points* | *-42.85 points* |
| **Setting C: Short Texts ($<50$ words)** | 96.55% Acc / 0.9114 F1 | 96.55% Acc / 0.9299 F1 | **98.85% Acc / 0.9729 F1** |
| **Setting C: Long Texts ($\ge 50$ words)** | 93.41% Acc / 0.9340 F1 | 93.67% Acc / 0.9367 F1 | **98.80% Acc / 0.9880 F1** |
| **Setting D: Hinglish WhatsApp ($N=60$)** | 55.00% Acc / 0.4555 F1 | 56.67% Acc / 0.4841 F1 | **60.00% Acc / 0.5238 F1** |
| **Expected Calibration Error (ECE)** | 2.09% | 4.24% | **0.48%** |
| **Abstention Accuracy Gain ($[0.4, 0.6]$)** | **+1.35%** (94.90% covered) | **+0.45%** (94.25% covered) | **+0.15%** (98.95% covered) |

### Key Takeaway
High benchmark accuracy does not translate into real-world generalization. When transferred from news articles to short political claims, **all models experience a >40-point drop in Macro-F1**. DistilBERT achieves the best zero-shot transfer on Hinglish forwards (60% accuracy, 0.8511 ROC-AUC, 100% fake recall) through subword tokenization, but models ultimately act as stylistic filters rather than factual fact-checkers.

---

## 🗂️ Repository Structure

```
fake-news-dl/
├── README.md                          # Project documentation and reproduction guide
├── requirements.txt                   # Pinned environment dependencies
├── config.yaml                        # Central configuration (paths, hyperparameters, hardware)
├── data/
│   ├── raw/                           # Raw cached downloads
│   ├── processed/                     # Cleaned, leakage-free CSVs (WELFake, LIAR, Pilot)
│   └── custom/
│       ├── pilot_author_written.csv   # Author-written pilot benchmark (N=60, balanced)
│       └── hinglish_forwards_template.csv # User-extensible CSV template
├── src/
│   ├── __init__.py
│   ├── utils.py                       # Seed setting, hardware auto-detection (MPS/CPU), logging
│   ├── preprocess.py                  # Wire leakage cleaning, deduplication, length tagging
│   ├── data.py                        # Dataset loading, schema normalization, stratified splits
│   ├── baseline.py                    # TF-IDF + Logistic Regression with C-hyperparameter tuning
│   ├── bilstm.py                      # SimpleVocab, BiLSTM architecture, scheduler, augmentation
│   ├── bert.py                        # DistilBERT fast pre-tokenization, training & inference
│   ├── evaluate.py                    # Metrics, confusion matrices, failure case extraction
│   ├── calibrate.py                   # ECE, reliability diagrams, abstention evaluation
│   └── explain.py                     # LIME explainability, HTML reports, attribution bar plots
├── scripts/
│   ├── download_data.py               # Data preparation and EDA visualization generator
│   ├── train_bilstm.py                # Dedicated BiLSTM training & sweep runner
│   ├── train_bert.py                  # Dedicated DistilBERT fine-tuning & sweep runner
│   ├── evaluate_all.py                # Comprehensive multi-setting evaluation runner
│   ├── generate_docx.py               # Converter from REPORT.md to formatted REPORT.docx
│   └── run_all.py                     # Unified master runner (supports --fast flag)
├── app/
│   └── app.py                         # Gradio web application with abstention & highlighting
├── models/                            # Saved model checkpoints and vocabularies
│   ├── baseline_model.joblib          # Canonical TF-IDF + LogReg pipeline
│   ├── bilstm_model.pt                # Canonical BiLSTM PyTorch weights
│   ├── bilstm_vocab.json              # Canonical BiLSTM vocabulary dictionary
│   └── distilbert_model/              # Canonical fine-tuned DistilBERT transformer
├── results/
│   ├── figures/                       # EDA, learning curves, reliability, and LIME PNGs
│   ├── tables/                        # experiments.csv, model_comparison.csv, calibration.csv
│   ├── failure_cases/                 # Categorized top-30 failure cases per model
│   └── metrics/                       # final_metrics.json
├── report/
│   ├── REPORT.md                      # Comprehensive academic project report (Markdown)
│   └── REPORT.docx                    # Formatted Word submission report
├── docs/
│   └── DECISIONS.md                   # Engineering, architectural, and design log
└── tests/
    └── test_pipeline.py               # Smoke & unit test suite (pytest)
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/fake-news-dl.git
cd fake-news-dl

# Create and activate virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Run Smoke Tests
Ensure all pipeline components, tensor forward passes, and app classification functions are sound:
```bash
pytest tests/
```

### 3. One-Command Master Evaluation
To re-evaluate all trained models across Settings A, B, C, D, generate figures, reliability diagrams, and compile metrics:
```bash
python scripts/evaluate_all.py
```
*(For a rapid 30-second dry run on tiny subsets: `python scripts/run_all.py --fast`)*

### 4. Generate Academic Report in Microsoft Word Format
```bash
python scripts/generate_docx.py
```
Outputs `report/REPORT.docx` with custom typography, headers, alternating-tint tables, and indentation.

### 5. Launch Interactive Gradio Web App
```bash
python app/app.py
```
Open your browser at `http://127.0.0.1:7860`. Features a high-fashion, ultra-clean editorial user interface (inspired by luxury minimalist design / GAZU aesthetic) with:
- Monochromatic obsidian & warm studio stone palette (`#F4F4F2` / `#0A0A0A`)
- Dark inverted architecture cards for model selection
- Real-time confidence visualization, abstention flagging, and token attribution
- Bottom empirical metric ribbon displaying benchmark accuracy, domain drop, and calibration metrics

---

## 🔬 Experimental Reproducibility & Integrity
- **Non-Negotiable Honesty**: Every metric reported in `report/REPORT.md` and this README is generated directly by code executed in this workspace. No synthetic or extrapolated numbers are used.
- **Hardware Agnostic**: Hardware backend is automatically detected (`mps` on Apple Silicon, `cuda` on Nvidia GPUs, or multi-threaded `cpu`).
- **Reproducible Seeds**: `random.seed(42)`, `np.random.seed(42)`, and `torch.manual_seed(42)` are fixed across all runs.

---

## ⚖️ Ethical Considerations & Screening Disclaimer

> **⚠️ MANDATORY DISCLAIMER:**  
> This system is designed solely as a **screening and decision-support aid** evaluating stylistic, structural, and linguistic resemblance to known misinformation patterns. **It does NOT verify factual truth or consult live ground-truth databases.** Critical news should always be verified against accredited primary news services and professional fact-checking organizations.
