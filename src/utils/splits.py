"""Helper utilities for working with train/val/test splits.

Most of the split logic lives in src.data.preprocessing.preprocess_clinical_and_splits,
which creates `splits.json`. This module can hold convenience functions if needed.
"""
import json
from typing import Dict

from src import config


def load_splits() -> Dict[str, list]:
    with open(config.SPLITS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)
