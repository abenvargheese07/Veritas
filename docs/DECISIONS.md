# Project Architecture & Engineering Decisions Log (docs/DECISIONS.md)

Project: **Fake News Detection Using Deep Learning: A Comparative Study of Classical ML, BiLSTM, and DistilBERT on Real-World, Hinglish-Aware News and Forwards**

---

### Decision 1: Python Runtime & Virtual Environment
- **Context**: The host machine had system Python 3.14.0, which lacks compatible wheels for some ML packages (e.g., PyTorch, scikit-learn). Homebrew Python 3.11 (`/opt/homebrew/bin/python3.11`) was detected.
- **Decision**: Created `.venv` using Python 3.11.14. Pinned all dependencies in `requirements.txt`.
- **Hardware Acceleration**: Detected Apple Silicon M3 with PyTorch MPS (Metal Performance Shaders) backend (`torch.backends.mps.is_available() == True`). Fallback to CPU is explicitly coded into runtime device selection.

### Decision 2: Workspace & Repository Directory Structure
- **Context**: The project workspace root is `/Users/abenvargheese/Documents/Code_Development/Projects/ANN`. The specification describes `fake-news-dl/`.
- **Decision**: Standard repository layout (`src/`, `scripts/`, `data/`, `models/`, `results/`, `app/`, `report/`, `docs/`, `tests/`) is initialized directly in the project workspace, and a symlink `fake-news-dl -> .` is created so that invocations from either current directory work identically without path breakage.

### Decision 3: Independent Datasets & Binary Label Mapping
- **Context**: Need at least two independent English fake news sources to test cross-dataset domain transfer (Setting B), plus short vs. long differences (Setting C).
- **Decision**:
  - **Dataset 1 (Short-form claims / Political fact-checks)**: **LIAR** (`liar` on Hugging Face / Hugging Face mirrors). Labels in LIAR: `pants-fire`, `false`, `barely-true` -> **1 (fake)**; `half-true`, `mostly-true`, `true` -> **0 (real)**. Documented standard convention in LIAR research literature (Wang, ACL 2017).
  - **Dataset 2 (Long-form articles / Full news stories)**: **WELFake** or **ISOT** (e.g. `mrm8488/fake-news` or Kaggle/ISOT mirror on Hugging Face). 0 = real news, 1 = fake news.
  - **Custom Dataset**: `data/custom/pilot_author_written.csv` (60 short WhatsApp forwards & Hinglish claims, balanced 30 real / 30 fake) and `data/custom/hinglish_forwards_template.csv` for extensibility.
  - **Schema**: `['text', 'label', 'source', 'length_type']`.

### Decision 4: Source Leakage Elimination
- **Context**: Many fake news datasets suffer from severe publisher/wire leakage (e.g., real news from Reuters always starts with `WASHINGTON (Reuters) - ...` or `(Reuters)`, while fake news from satirical or blog sites includes specific author names or URL trails). If uncleaned, models achieve ~99% accuracy simply by memorizing wire stamps.
- **Decision**: Implemented regex and prefix stripping in `src/preprocess.py` to remove wire service markers (`Reuters`, `Associated Press`, `AP`, `CNN`, etc.), URLs, email addresses, bracketed location headers (`[City]`), and repeated punctuation.

### Decision 5: Calibration & Abstention Policy
- **Context**: Fake news classifiers in the wild frequently make overconfident errors on out-of-domain text or informal slang.
- **Decision**: Added expected calibration error (ECE) measurement, reliability diagrams, and a confidence band abstention mechanism. If `0.40 <= P(fake) <= 0.60`, the model outputs "Not sure / Inconclusive" rather than forcing a low-confidence binary decision.

### Decision 6: Explainability with LIME & Token Attribution
- **Context**: The examiner requires understanding whether models learn semantic veracity or superficial stylistic cues.
- **Decision**: LIME Text Explainer is integrated for all models, highlighting positive and negative word importances. HTML and visual bar representations are saved in `results/figures/` and integrated into the Gradio web UI.

### Decision 7: Execution Pause & Hold State
- **Context**: The user requested an immediate stop and hold on ongoing runs with a persistent task inventory.
- **Decision**: Terminated all active processes (including `run_all.py` process 19715). Created `docs/ON_HOLD_TASKS.md` detailing all completed components, artifacts, and exact steps/commands to resume.

### Decision 8: UI Redesign & High-Fashion Editorial Aesthetic (GAZU Palette)
- **Context**: The user requested a complete visual overhaul of the web frontend (`app/app.py`), providing a high-fashion editorial reference ("GAZU") featuring an ultra-clean, minimalist, architectural monochrome palette.
- **Decision**:
  - Replaced generic Gradio styling with custom CSS and a tailored `gr.themes.Base` theme.
  - Palette tokens:
    - Primary stone canvas: `#F4F4F2` (warm studio neutral background)
    - Obsidian dark accents: `#0A0A0A` / `#0D0D0D` (contrast header cards, solid buttons)
    - Hairline architectural borders: `#DCDCDA` / `#262626` (1px clean dividers)
    - Typography: Grotesque sans-serif (`Inter`, `Helvetica Neue`), uppercase tracking (`letter-spacing: 0.15em - 0.25em`), bold headline scale (`VERITAS`).
  - Layout Architecture:
    - Top minimalist utility banner (`ACADEMIC DEEP LEARNING AUDIT`, hardware engine status, seed reproducibility).
    - Giant editorial hero banner (`VERITAS`).
    - 3-card dark inverted model category strip (`TRANSFORMER`, `RECURRENT DL`, `CLASSICAL ML`).
    - Crisp dual-column workspace (`01 / EMPIRICAL VERACITY INPUT` and `02 / EMPIRICAL SCREENING DIAGNOSTICS`).
    - 4-column metric ribbon displaying real empirical benchmarks (WELFake In-domain F1, LIAR domain shift drop, ECE, Abstention coverage).
  - Maintained complete backwards compatibility with all automated smoke tests (`classify_text` 4-tuple signature).


