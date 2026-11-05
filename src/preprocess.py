"""Text Preprocessing and Leakage Removal Module.

Contains cleaning functions to strip publisher wire leakage, URLs, metadata stamps,
and normalize text across independent datasets.
"""

import re
from typing import Dict, List, Tuple
import pandas as pd


# Regex patterns for source and wire leakage
WIRE_PATTERNS = [
    # Location + Wire combinations e.g. "WASHINGTON (Reuters) -", "MOSCOW (Reuters) -"
    r"^[A-Z\s,./-]{2,40}\s*\((?:Reuters|REUTERS|AP|AFP|Bloomberg)\)\s*[-—:]\s*",
    r"^\((?:Reuters|REUTERS|AP|AFP|Bloomberg)\)\s*[-—:]\s*",
    r"^\[(?:Reuters|REUTERS|AP|AFP|Bloomberg)\]\s*[-—:]\s*",
    r"^By\s+[\w\s.-]{2,40}\s*\|\s*(?:Reuters|AP|Bloomberg|CNN|Breitbart)\s*[-—:]?\s*",
    r"\(Reuters\)",
    r"\[Reuters\]",
    r"\(AP\)",
    r"\[AP\]",
    r"\(AFP\)",
    r"\[AFP\]",
    r"via\s+(?:Breitbart|Infowars|The Daily Caller|Natural News)",
    r"(?:Reporting|Writing)\s+by\s+[\w\s.,-]{2,50}(?:;?\s*(?:editing|Editing)\s+by\s+[\w\s.,-]{2,50})?\.?",
    r"Featured\s+image\s+via\s+[\w\s.,-]{2,50}",
]

URL_PATTERN = r"https?://\S+|www\.\S+|bit\.ly/\S+"
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
HTML_TAGS = r"<[^>]+>"


def clean_leakage_and_noise(text: str) -> str:
    """Strip obvious publisher leakage, wires, URLs, and noisy artifacts."""
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()

    # Strip HTML tags
    cleaned = re.sub(HTML_TAGS, " ", cleaned)

    # Strip URLs and Emails
    cleaned = re.sub(URL_PATTERN, " ", cleaned)
    cleaned = re.sub(EMAIL_PATTERN, " ", cleaned)

    # Strip publisher / wire leakage patterns
    for pat in WIRE_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

    # Clean non-standard unicode whitespace and multiple spaces
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def assign_length_type(text: str, threshold: int = 50) -> str:
    """Classify text length bucket: 'short' (< 50 words) or 'long' (>= 50 words)."""
    words = text.split()
    return "short" if len(words) < threshold else "long"


def remove_duplicates_and_empty(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Drop empty rows, exact duplicates, and normalized near-duplicates."""
    initial_len = len(df)
    # Drop na and empty
    df = df.dropna(subset=[text_col]).copy()
    df["clean_temp"] = df[text_col].astype(str).str.strip().str.lower()
    df = df[df["clean_temp"].str.len() > 10]

    # Deduplicate on normalized text
    df = df.drop_duplicates(subset=["clean_temp"]).drop(columns=["clean_temp"])
    final_len = len(df)
    return df


def demonstrate_leakage_cleaning() -> Dict[str, str]:
    """Provide a reproducible before-and-after demonstration of wire leakage removal."""
    sample_raw = (
        "WASHINGTON (Reuters) - The U.S. Senate passed a landmark bipartisan infrastructure "
        "package on Tuesday afternoon. Read full details at https://reuters.com/article/12345. "
        "Reporting by David Morgan; Editing by Sonya Hepinstall."
    )
    sample_clean = clean_leakage_and_noise(sample_raw)
    return {
        "before": sample_raw,
        "after": sample_clean,
        "explanation": (
            "The raw text contained the wire header 'WASHINGTON (Reuters) -', an embedded "
            "canonical URL, and editorial reporter signatures ('Reporting by...'). Without "
            "cleaning, an NLP model memorizes 'Reuters' or 'Reporting by' as a 100% shortcut "
            "for 'real news', inflating cross-validation accuracy artificially."
        )
    }
