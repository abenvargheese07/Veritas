"""Utility functions for Fake News DL project.

Provides reproducible seeding, device management, configuration loading,
logging, and filesystem helpers.
"""

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS does not currently support deterministic algorithms for all ops,
        # but manual_seed ensures repeatable pseudo-random generator state.
        pass


def get_device(preference: str = "auto") -> torch.device:
    """Detect and return the best available compute device.

    Supports Apple Silicon MPS, NVIDIA CUDA, and CPU fallback.
    """
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")

    # Auto mode
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        # Fallback to repo root if relative path differs
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name: str = "fake_news_dl", log_file: str = None) -> logging.Logger:
    """Set up and return a formatted console and optional file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger
