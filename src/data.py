"""Data Loading, Normalization, Splitting, and Persistence Module.

Handles loading raw datasets from Hugging Face / custom sources, normalizes them
to the standard schema ['text', 'label', 'source', 'length_type'], cleans leakage,
and creates stratified splits.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from src.preprocess import clean_leakage_and_noise, assign_length_type, remove_duplicates_and_empty
from src.utils import setup_logger

logger = setup_logger("data_loader")


def load_liar_dataset(
    max_samples: Optional[int] = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalize the LIAR dataset (short-form political statements).

    Label mapping:
      0 (pants-fire), 1 (false), 2 (barely-true) -> 1 (Fake)
      3 (half-true), 4 (mostly-true), 5 (true) -> 0 (Real)
    """
    logger.info("Loading LIAR dataset from Hugging Face (chengxuphd/liar2)...")
    splits = {}
    split_map = {"train": "train", "val": "validation", "test": "test"}
    for out_split, hf_split in split_map.items():
        ds = load_dataset("chengxuphd/liar2", split=hf_split)
        df = ds.to_pandas()

        # Binary label mapping: {0,1,2} -> 1 (fake), {3,4,5} -> 0 (real)
        df["binary_label"] = df["label"].apply(lambda x: 1 if x in [0, 1, 2] else 0)

        # Standard text column
        df["text"] = df["statement"].apply(clean_leakage_and_noise)
        df["label"] = df["binary_label"]
        df["source"] = "liar"
        df["length_type"] = df["text"].apply(assign_length_type)

        df = df[["text", "label", "source", "length_type"]]
        df = remove_duplicates_and_empty(df)
        splits[out_split] = df

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    if max_samples and len(train_df) > max_samples:
        parts = []
        n_per_class = max_samples // 2
        for _, group in train_df.groupby("label"):
            n = min(len(group), n_per_class)
            parts.append(group.sample(n=n, random_state=seed))
        train_df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(
        f"LIAR loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df


def load_welfake_dataset(
    max_train: int = 12000,
    max_val: int = 2000,
    max_test: int = 2000,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalize the WELFake dataset (article-level long texts).

    0 = Real, 1 = Fake.
    """
    logger.info("Loading WELFake dataset from Hugging Face (davanstrien/WELFake)...")
    ds = load_dataset("davanstrien/WELFake", split="train")
    df = ds.to_pandas()

    # Drop null labels or text
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Prefer text; if empty/short use title
    title_str = df["title"].fillna("").astype(str)
    text_str = df["text"].fillna("").astype(str)
    combined_text = title_str + ". " + text_str

    df["text"] = combined_text.apply(clean_leakage_and_noise)
    df["source"] = "welfake"
    df["length_type"] = df["text"].apply(assign_length_type)

    df = df[["text", "label", "source", "length_type"]]
    df = remove_duplicates_and_empty(df)

    # Stratified split: train (80%), temp (20%) -> val (10%), test (10%)
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"]
    )

    def balance_subsample(dataframe: pd.DataFrame, target_size: int) -> pd.DataFrame:
        if len(dataframe) <= target_size:
            return dataframe
        n_per_class = target_size // 2
        parts = []
        for _, group in dataframe.groupby("label"):
            n = min(len(group), n_per_class)
            parts.append(group.sample(n=n, random_state=seed))
        return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df = balance_subsample(train_df, max_train)
    val_df = balance_subsample(val_df, max_val)
    test_df = balance_subsample(test_df, max_test)

    logger.info(
        f"WELFake prepared: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_custom_pilot_data(
    file_path: str = "data/custom/pilot_author_written.csv"
) -> pd.DataFrame:
    """Load the author-written pilot set of WhatsApp forwards and Hinglish items."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom pilot data not found at {path}")

    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    if "source" not in df.columns:
        df["source"] = "pilot_author_written"
    if "length_type" not in df.columns:
        df["length_type"] = df["text"].apply(assign_length_type)

    return df[["text", "label", "source", "length_type"]].reset_index(drop=True)


def prepare_and_save_all_data(
    output_dir: str = "data/processed",
    max_train_samples: int = 12000,
    max_val_samples: int = 2000,
    max_test_samples: int = 2000,
    fast_mode: bool = False,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """Execute end-to-end dataset preparation and save processed CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    if fast_mode:
        max_train = 500
        max_val = 100
        max_test = 100
    else:
        max_train = max_train_samples
        max_val = max_val_samples
        max_test = max_test_samples

    # Load LIAR
    liar_train, liar_val, liar_test = load_liar_dataset(
        max_samples=max_train, seed=seed
    )
    if fast_mode:
        liar_val = liar_val.sample(n=min(len(liar_val), max_val), random_state=seed)
        liar_test = liar_test.sample(n=min(len(liar_test), max_test), random_state=seed)

    # Load WELFake
    welfake_train, welfake_val, welfake_test = load_welfake_dataset(
        max_train=max_train, max_val=max_val, max_test=max_test, seed=seed
    )

    # Load Custom Pilot
    custom_pilot = load_custom_pilot_data()

    # Save to disk
    datasets_dict = {
        "liar_train": liar_train,
        "liar_val": liar_val,
        "liar_test": liar_test,
        "welfake_train": welfake_train,
        "welfake_val": welfake_val,
        "welfake_test": welfake_test,
        "pilot_custom": custom_pilot,
    }

    for name, df in datasets_dict.items():
        save_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(save_path, index=False)
        logger.info(f"Saved {name} ({len(df)} rows) to {save_path}")

    return datasets_dict
