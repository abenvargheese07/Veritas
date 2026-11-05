# Project Status & Remaining Tasks on Hold (docs/ON_HOLD_TASKS.md)

**Project Title:** Fake News Detection Using Deep Learning: A Comparative Study of Classical ML, BiLSTM, and DistilBERT on Real-World, Hinglish-Aware News and Forwards  
**Date Paused:** November 4, 2026  
**Status:** **ON HOLD / PAUSED AT USER REQUEST**  
**All running background processes:** Fully halted (`kill` signal delivered to `run_all.py` process 19715; verified stopped).

---

## 1. What Has Been Completed & Verified

| Component | Status | Details / Artifacts Created |
| :--- | :---: | :--- |
| **Virtual Environment & Dependencies** |  Completed | Python 3.11 `.venv` created; Apple Silicon M3 MPS acceleration detected & configured; all dependencies pinned in `requirements.txt`. |
| **Repository Structure** |  Completed | Standard modular directory structure initialized (`src/`, `scripts/`, `data/`, `models/`, `results/`, `app/`, `report/`, `docs/`, `tests/`) with root symlink `fake-news-dl -> .`. |
| **Leakage Stripping & Normalization** |  Completed | `src/preprocess.py` implements wire marker stripping (Reuters, AP, AFP, CNN), reporter signatures, URLs, emails, and near-duplicate filtering. Before/after demo verified. |
| **Datasets Downloaded & Schema Normalization** |  Completed | LIAR (short political claims) and WELFake (long article news) normalized to `['text', 'label', 'source', 'length_type']` with binary mapping `0=real, 1=fake`. |
| **Custom Hinglish & WhatsApp Pilot** |  Completed | `data/custom/pilot_author_written.csv` created (60 balanced rows: 30 real, 30 fake, Hinglish & English forwards) and `data/custom/hinglish_forwards_template.csv` template. |
| **Exploratory Data Analysis (EDA)** |  Completed | `scripts/download_data.py` executed; plots saved: `results/figures/eda_class_balance.png`, `eda_length_distribution.png`, `eda_top_words.png`. |
| **Model Architectures Implemented** |  Completed | `src/baseline.py` (TF-IDF + LogisticRegression with CV tuning), `src/bilstm.py` (Tokenizer, Embedding, BiLSTM, Dropout, LR scheduler, Grad clip, Augmentation), `src/bert.py` (DistilBERT fine-tuning). |
| **Evaluation Framework** |  Completed | `src/evaluate.py` implements Settings A (In-dataset), B (Cross-dataset domain shift), C (Short vs. Long), D (Hinglish/WhatsApp pilot). Confusion matrices and top-30 failure case extraction. |
| **Calibration & Abstention** |  Completed | `src/calibrate.py` implements ECE, reliability diagrams, and abstention rule (`0.40 <= P(fake) <= 0.60`). |
| **Explainability (LIME)** |  Completed | `src/explain.py` generates interactive HTML & PNG bar charts for word importance attribution. |
| **Gradio Web Application** |  Completed | `app/app.py` created with paste box, model selector, abstention ("Not sure / Inconclusive"), word cue highlights, and screening disclaimer. |
| **Smoke Tests** |  Completed | `tests/test_pipeline.py` written and verified passing (`5 passed in 2.76s` via pytest). |
| **End-to-End Pipeline Script** |  Completed | `scripts/run_all.py` created and tested in `--fast` mode (verified 100% working without exceptions). |

---

## 2. What Is Left To Do (When Resuming)

The following 4 sequential steps remain to fulfill the full Definition of Done:

### Task 1: Complete Full Production Model Training Run
- **Command:** `source .venv/bin/activate && python scripts/run_all.py`
- **What it does:**
  1. Trains Baseline TF-IDF + Logistic Regression on full 12,000 WELFake samples (C-regularization tuned).
  2. Runs 4 BiLSTM experiments (Base hidden=128, Var1 hidden=64, Var2 drop=0.5, Augmentation with word dropout/swap).
  3. Runs 3 DistilBERT fine-tuning experiments on 8,000 samples with Apple M3 MPS (Base lr=2e-5, Var1 lr=5e-5, Var2 max_length=64).
  4. Automatically records all runs in `results/tables/experiments.csv`.
  5. Computes metrics across Settings A, B, C, D and saves `results/tables/model_comparison.csv` and `results/figures/model_comparison.png`.
  6. Generates reliability diagrams, saves `results/tables/calibration_results.csv`, extracts top-30 failure cases per model to `results/failure_cases/`, and produces LIME case studies.
- **Estimated Runtime:** ~8–12 minutes on Apple Silicon M3.

### Task 2: Verify & Smoke-Test Gradio App
- **Action:** Test `app/app.py` with saved production weights by executing a programmatic test call or launching `python app/app.py`.
- **Verify:** Empty input handling, long input truncation (>5,000 chars), Hinglish forward input, and abstention band rendering.

### Task 3: Generate Academic Report (`REPORT.md` & `REPORT.docx`)
- **Location:** `report/REPORT.md` and `report/REPORT.docx` (via `python-docx`).
- **Required Sections & Evidence:**
  - Title page placeholders: `[YOUR NAME]`, `[YOUR ROLL NO]`, `[COURSE]`, `[UNIVERSITY]`
  - Abstract & Motivation (addressing the ~99% benchmark illusion vs. real-world generalization failure)
  - Research Questions & Hypotheses
  - Datasets & Preprocessing (including before/after wire leakage table)
  - Methodology & Neural Architectures (including ASCII/Mermaid architecture diagrams)
  - Experimental Setup (Hardware Apple Silicon M3, MPS, random seeds = 42, hyperparameters)
  - Results Section with exact numbers from `results/tables/model_comparison.csv`:
    * Setting A: In-Dataset performance
    * Setting B: Cross-Dataset domain shift drop
    * Setting C: Short vs. Long text performance
    * Setting D: Hinglish/WhatsApp pilot set performance (sample size limitations noted)
  - Calibration & Abstention (ECE numbers, coverage vs. accuracy trade-off)
  - Failure Analysis (30 most confident wrong predictions per model categorized by observed patterns)
  - Explainability Discussion (LIME case studies: semantic vs. superficial/stylistic cues)
  - Limitations, Ethical Considerations, Fact-checking distinction
  - Real, verified academic references.

### Task 4: Complete `README.md`
- **Location:** `README.md`
- **Contents:** Setup instructions (`requirements.txt`), one-command reproduction (`python scripts/run_all.py`), quick sanity run (`python scripts/run_all.py --fast`), app launch (`python app/app.py`), repository tour, and key empirical takeaways.

---

## 3. Exact Commands to Resume When Ready

To resume immediately from this checkpoint:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run full production training and evaluation
python scripts/run_all.py

# 3. Launch and test the Gradio application
python app/app.py
```
