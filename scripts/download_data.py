#!/usr/bin/env python3
"""Data Download, Preprocessing, and Exploratory Data Analysis (EDA) Script.

Downloads LIAR and WELFake datasets, integrates custom Hinglish/pilot forwards,
applies publisher wire leakage cleaning, saves processed splits, and produces
EDA figures in results/figures/.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
import re

from src.data import prepare_and_save_all_data, load_custom_pilot_data
from src.preprocess import demonstrate_leakage_cleaning
from src.utils import set_seed, setup_logger, load_config

logger = setup_logger("download_data")

# Minimal English stop words for EDA frequency inspection
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves", "said", "says", "one", "also", "would", "new", "us"
}


def run_eda(datasets: dict, output_dir: str = "results/figures") -> None:
    """Generate exploratory data analysis visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    liar_train = datasets["liar_train"]
    welfake_train = datasets["welfake_train"]
    pilot = datasets["pilot_custom"]

    # 1. Class Balance Bar Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (name, df) in zip(axes, [("LIAR (Short)", liar_train), ("WELFake (Long)", welfake_train), ("Hinglish Pilot", pilot)]):
        counts = df["label"].value_counts().rename(index={0: "Real (0)", 1: "Fake (1)"})
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=["#2ecc71", "#e74c3c"])
        ax.set_title(f"{name} Class Balance (Total={len(df)})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_xlabel("Label")
        for i, v in enumerate(counts.values):
            ax.text(i, v + (max(counts.values) * 0.02), f"{v} ({v/len(df):.1%})", ha="center", fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "eda_class_balance.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logger.info(f"Saved EDA class balance plot to {fig_path}")

    # 2. Text Length Distribution (Word Count)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LIAR word length
    liar_words_real = liar_train[liar_train["label"] == 0]["text"].apply(lambda t: len(t.split()))
    liar_words_fake = liar_train[liar_train["label"] == 1]["text"].apply(lambda t: len(t.split()))

    axes[0].hist(liar_words_real, bins=30, alpha=0.6, label="Real (0)", color="#2ecc71", density=True)
    axes[0].hist(liar_words_fake, bins=30, alpha=0.6, label="Fake (1)", color="#e74c3c", density=True)
    axes[0].set_title("LIAR Word Count Distribution (Short Text)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(0, 100)
    axes[0].legend()

    # WELFake word length (clipped for visualization)
    welfake_words_real = welfake_train[welfake_train["label"] == 0]["text"].apply(lambda t: len(t.split()))
    welfake_words_fake = welfake_train[welfake_train["label"] == 1]["text"].apply(lambda t: len(t.split()))

    axes[1].hist(np.clip(welfake_words_real, 0, 1500), bins=35, alpha=0.6, label="Real (0)", color="#2ecc71", density=True)
    axes[1].hist(np.clip(welfake_words_fake, 0, 1500), bins=35, alpha=0.6, label="Fake (1)", color="#e74c3c", density=True)
    axes[1].set_title("WELFake Word Count Distribution (Long Text)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Word Count (Clipped at 1500 words)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "eda_length_distribution.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logger.info(f"Saved EDA length distribution plot to {fig_path}")

    # 3. Top Words per Class (WELFake & LIAR)
    def get_top_tokens(series: pd.Series, top_k: int = 15) -> list:
        words = []
        for text in series:
            tokens = re.findall(r"\b[a-zA-Z]{3,15}\b", text.lower())
            words.extend([w for w in tokens if w not in STOP_WORDS])
        return Counter(words).most_common(top_k)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # LIAR top words
    liar_real_top = get_top_tokens(liar_train[liar_train["label"] == 0]["text"])
    liar_fake_top = get_top_tokens(liar_train[liar_train["label"] == 1]["text"])

    sns.barplot(x=[c for _, c in liar_real_top], y=[w for w, _ in liar_real_top], ax=axes[0, 0], color="#2ecc71")
    axes[0, 0].set_title("LIAR: Top Words in Real Claims", fontweight="bold")
    axes[0, 0].set_xlabel("Frequency")

    sns.barplot(x=[c for _, c in liar_fake_top], y=[w for w, _ in liar_fake_top], ax=axes[0, 1], color="#e74c3c")
    axes[0, 1].set_title("LIAR: Top Words in Fake Claims", fontweight="bold")
    axes[0, 1].set_xlabel("Frequency")

    # WELFake top words
    wf_real_top = get_top_tokens(welfake_train[welfake_train["label"] == 0]["text"])
    wf_fake_top = get_top_tokens(welfake_train[welfake_train["label"] == 1]["text"])

    sns.barplot(x=[c for _, c in wf_real_top], y=[w for w, _ in wf_real_top], ax=axes[1, 0], color="#2ecc71")
    axes[1, 0].set_title("WELFake: Top Words in Real News", fontweight="bold")
    axes[1, 0].set_xlabel("Frequency")

    sns.barplot(x=[c for _, c in wf_fake_top], y=[w for w, _ in wf_fake_top], ax=axes[1, 1], color="#e74c3c")
    axes[1, 1].set_title("WELFake: Top Words in Fake News", fontweight="bold")
    axes[1, 1].set_xlabel("Frequency")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "eda_top_words.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    logger.info(f"Saved EDA top words plot to {fig_path}")

    # 4. Leakage Verification Check
    logger.info("Executing leakage audit on processed datasets...")
    leakage_terms = ["(reuters)", "reuters -", "associated press", "via breitbart"]
    for name, df in [("LIAR Train", liar_train), ("WELFake Train", welfake_train)]:
        for term in leakage_terms:
            matches = df["text"].str.lower().str.contains(term, regex=False).sum()
            logger.info(f"Leakage audit: '{term}' in {name} = {matches} occurrences")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare fake news datasets.")
    parser.add_argument("--fast", action="store_true", help="Quick mode for testing with small subsets.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["random_seed"])

    logger.info(f"Starting data preparation (Fast Mode={args.fast})...")

    # Leakage demo logging
    demo = demonstrate_leakage_cleaning()
    logger.info(f"Leakage cleaning demonstration:\nBEFORE: {demo['before']}\nAFTER:  {demo['after']}")

    datasets = prepare_and_save_all_data(
        output_dir=cfg["paths"]["processed_dir"],
        max_train_samples=cfg["data"]["max_train_samples_per_dataset"],
        max_val_samples=cfg["data"]["max_val_samples_per_dataset"],
        max_test_samples=cfg["data"]["max_test_samples_per_dataset"],
        fast_mode=args.fast,
        seed=cfg["project"]["random_seed"]
    )

    logger.info("Running Exploratory Data Analysis (EDA)...")
    run_eda(datasets, output_dir=cfg["paths"]["figures_dir"])
    logger.info("Data preparation & EDA completed successfully!")


if __name__ == "__main__":
    main()
