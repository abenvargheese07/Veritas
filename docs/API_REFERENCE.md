# Python API & Module Reference

**Project:** VERITAS — Fake News & Forward Veracity Screening Platform  
**Package:** `src/`  
**Python Version:** $\ge 3.10$

---

## Table of Contents
1. [`src.utils`](#1-srcutils)
2. [`src.preprocess`](#2-srcpreprocess)
3. [`src.data`](#3-srcdata)
4. [`src.baseline`](#4-srcbaseline)
5. [`src.bilstm`](#5-srcbilstm)
6. [`src.bert`](#6-srcbert)
7. [`src.evaluate`](#7-srcevaluate)
8. [`src.calibrate`](#8-srccalibrate)
9. [`src.explain`](#9-srcexplain)
10. [`scripts/` CLI Interface](#10-scripts-cli-interface)

---

## 1. `src.utils`
Utility functions for hardware detection, reproducibility seeds, and configuration loading.

### `get_device() -> torch.device`
Detects and returns the best available PyTorch compute device:
- Returns `torch.device("mps")` if Apple Silicon Metal Performance Shaders are available.
- Returns `torch.device("cuda")` if an NVIDIA GPU is detected.
- Returns `torch.device("cpu")` as fallback.

### `set_seed(seed: int = 42) -> None`
Fixes pseudorandom number generator seeds across `random`, `numpy`, `os.environ["PYTHONHASHSEED"]`, `torch`, and backend hardware generators (`torch.cuda` / `torch.mps`) to ensure 100% deterministic reproducibility.

### `setup_logger(name: str = "veritas", level: int = logging.INFO) -> logging.Logger`
Configures and returns a standardized terminal logger with timestamp and level formatting.

### `load_config(config_path: str = "config.yaml") -> Dict[str, Any]`
Safely parses and loads the project's YAML configuration file into a nested Python dictionary.

---

## 2. `src.preprocess`
Data cleaning and source leakage mitigation routines.

### `clean_wire_prefixes(text: str) -> str`
Strips leading wire service markers (e.g. `"WASHINGTON (Reuters) -"` or `"LONDON (AP) --"`) using regex pattern matching.

### `clean_leakage_and_noise(text: str) -> str`
Executes full sanitization on a single text string:
1. Strips wire agency prefixes.
2. Strips trailing reporter signatures (`"Reporting by... Editing by..."`).
3. Strips URLs (`http\S+`, `www\.\S+`) and email addresses.
4. Unescapes HTML entities (`&amp;`, `&quot;`).
5. Collapses redundant whitespace.

### `clean_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame`
Applies `clean_leakage_and_noise` across an entire pandas DataFrame column, drops empty/null rows, and removes near-duplicates based on normalized string hashing.

### `assign_length_type(text: str, threshold: int = 50) -> str`
Tokenizes text on whitespace. Returns `"short"` if word count is $< \text{threshold}$, else `"long"`.

---

## 3. `src.data`
Dataset loading, schema normalization, and stratified splitting.

### `normalize_schema(df: pd.DataFrame, source_name: str, length_threshold: int = 50) -> pd.DataFrame`
Standardizes raw datasets to the canonical 4-column schema:
- `text` (str): Cleaned news or forward text.
- `label` (int): Binary label (`0` = real/credible, `1` = fake/misinformation).
- `source` (str): Origin dataset identifier (e.g., `"WELFake"`, `"LIAR"`).
- `length_type` (str): Categorical length bucket (`"short"` vs. `"long"`).

### `prepare_and_split_data(...) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
Performs stratified train/validation/test splitting (`random_state=42`) according to ratios specified in `config.yaml`.

---

## 4. `src.baseline`
Classical Machine Learning Baseline (TF-IDF + Logistic Regression).

### `train_baseline_pipeline(X_train: List[str], y_train: List[int], C: float = 5.0) -> Pipeline`
Constructs and fits a `scikit-learn` Pipeline consisting of:
- `TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)`
- `LogisticRegression(C=C, max_iter=200, random_state=42)`

### `predict_baseline(pipeline: Pipeline, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]`
Returns:
- `preds` (np.ndarray): Binary class predictions ($0$ or $1$).
- `probs` (np.ndarray): Posterior probability of falsehood $P(Y=1 \mid X)$.

### `save_baseline_model(pipeline: Pipeline, path: str) -> None`
Serializes the fitted pipeline to disk using `joblib`.

### `load_baseline_model(path: str) -> Pipeline`
Loads and restores a serialized pipeline from disk.

---

## 5. `src.bilstm`
Recurrent Deep Learning architecture with word embeddings and data augmentation.

### `class SimpleVocab`
Custom word-level vocabulary builder:
- `build_vocab(texts: List[str], max_vocab: int = 25000, min_freq: int = 3)`
- `encode(tokens: List[str], max_len: int = 200) -> List[int]`
- Special tokens: `<pad>` (index 0), `<unk>` (index 1).

### `class BiLSTMClassifier(nn.Module)`
PyTorch neural network module:
- `nn.Embedding(vocab_size, embedding_dim=128, padding_idx=0)`
- `nn.LSTM(embedding_dim, hidden_dim=128, bidirectional=True, batch_first=True)`
- `nn.Dropout(p=0.30)`
- `nn.Linear(256, 2)`

### `train_bilstm_model(...) -> Tuple[BiLSTMClassifier, SimpleVocab, Dict[str, List[float]]]`
Executes end-to-end training loop with:
- Adam optimizer and `ReduceLROnPlateau` scheduler.
- Gradient norm clipping (`max_norm = 1.0`).
- Early stopping tracking validation Macro-F1.
- Optional stochastic word dropout / swap augmentation.

### `predict_bilstm(model, vocab, texts: List[str], device) -> Tuple[np.ndarray, np.ndarray]`
Tokenizes and runs forward pass on input texts, returning binary predictions and probabilities.

---

## 6. `src.bert`
Transformer fine-tuning via Hugging Face and PyTorch.

### `get_distilbert_model_and_tokenizer(model_name: str = "distilbert-base-uncased", num_labels: int = 2)`
Initializes pretrained `DistilBertForSequenceClassification` and `DistilBertTokenizerFast`.

### `train_distilbert(...) -> Tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast, Dict]`
Fine-tunes DistilBERT with:
- Pre-tokenized PyTorch TensorDatasets (`max_length = 128`).
- AdamW optimizer with weight decay exclusion on bias and LayerNorm.
- Linear warmup schedule (10% steps).
- Multi-threaded batch data loaders.

### `predict_distilbert(model, tokenizer, texts: List[str], device) -> Tuple[np.ndarray, np.ndarray]`
Runs batched forward pass with `torch.no_grad()`, returning class labels and softmax probabilities.

---

## 7. `src.evaluate`
Performance metrics, confusion matrices, and failure case extraction.

### `compute_classification_metrics(y_true, y_pred, y_prob) -> Dict[str, float]`
Computes:
- `accuracy`: Overall classification accuracy.
- `precision`: Macro and binary positive precision.
- `recall`: Macro and binary positive recall.
- `f1_macro`: Unweighted average of per-class F1-scores.
- `roc_auc`: Area under the Receiver Operating Characteristic curve.

### `extract_top_failures(df: pd.DataFrame, y_true, y_pred, y_prob, top_k: int = 30) -> pd.DataFrame`
Filters for misclassified instances ($y_{\text{true}} \neq y_{\text{pred}}$) and sorts by confidence distance $|P(\text{Fake}) - 0.5|$ to locate the most confident incorrect predictions.

---

## 8. `src.calibrate`
Probability calibration and abstention policy.

### `compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float`
Calculates the Expected Calibration Error across $n$ equal-width bins.

### `plot_reliability_diagram(y_true, y_prob, n_bins=10, save_path=None)`
Generates a reliability diagram comparing bin confidence against empirical accuracy alongside bin sample count bars.

### `apply_abstention_policy(y_true, y_prob, lower_thresh: float = 0.40, upper_thresh: float = 0.60)`
Applies the interval $[0.40, 0.60]$:
- Returns sample coverage $\frac{|\text{covered}|}{N}$.
- Returns accuracy over covered samples.
- Returns number and percentage of abstained items.

---

## 9. `src.explain`
Local Interpretable Model-agnostic Explanations (LIME).

### `generate_lime_explanation(text: str, predict_fn, class_names=["Real", "Fake"], num_features=8)`
Instantiates `LimeTextExplainer` and samples $K=200$ perturbations to output token attributions.

### `explain_samples_batch(samples, predict_fn, save_dir: str)`
Batch-generates HTML visualizers and PNG horizontal bar charts for curated case studies.

---

## 10. `scripts/` CLI Interface

| Command | Key Flags | Description |
| :--- | :--- | :--- |
| `python scripts/download_data.py` | `--data_dir` | Downloads WELFake & LIAR, cleans leakage, creates pilot dataset, and produces EDA charts. |
| `python scripts/train_bilstm.py` | `--epochs`, `--batch_size` | Runs BiLSTM baseline and hyperparameter sweep experiments. |
| `python scripts/train_bert.py` | `--epochs`, `--lr`, `--max_len` | Runs DistilBERT baseline and hyperparameter sweep experiments. |
| `python scripts/evaluate_all.py` | `--models_dir`, `--results_dir` | Evaluates all models across Settings A, B, C, D, calibration, and failure cases. |
| `python scripts/generate_docx.py` | `--md_path`, `--docx_path` | Converts `report/REPORT.md` into formatted `report/REPORT.docx`. |
| `python scripts/run_all.py` | `--fast` | Master runner; `--fast` runs a rapid 30-second dry run on sample subsets. |
| `python app/app.py` | `--port`, `--share` | Launches the interactive Gradio web application at `http://127.0.0.1:7860`. |
