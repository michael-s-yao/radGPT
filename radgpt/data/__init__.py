"""
Interface for working with datasets of patient one-liner case descriptions.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import pandas as pd
from typing import Optional

from .dataset import (
    convert_case_to_one_liner,
    hashme,
    read_synthetic_dataset,
    read_mimic_iv_dataset,
    read_jama_cc_dataset,
    read_medbullets_dataset,
    read_nejm_dataset
)
from . import utils


__all__ = [
    "convert_case_to_one_liner",
    "hashme",
    "read_mimic_iv_dataset",
    "read_nejm_dataset",
    "read_jama_cc_dataset",
    "read_medbullets_dataset",
    "read_synthetic_dataset",
    "load_case_labels",
    "utils"
]


def load_case_labels(
    dataset: str, fn_or_url: Optional[str] = None
) -> pd.DataFrame:
    """
    Loads the ground-truth patient case labels.
    Input:
        dataset: the dataset of patient one-liners to use.
        fn_or_url: an optional specified filepath or URL to load from instead.
    Returns:
        The ground-truth patient case labels.
    """
    cols = ["case", "panel", "topic", "topic_id"]

    if fn_or_url:
        return pd.read_csv(fn_or_url)[cols]

    dataset_to_gid = {
        "synthetic": 1839683815,
        "medbullets": 0,
        "jama_cc": 226315523,
        "mimic_iv": 42526063,
        "nejm": 1217489264,
    }
    labels_url: str = (
        "https://docs.google.com/spreadsheets/d/"
        "1PNu-rAbQG3SAAhQ7TZqOaS4cT7V8033dKVDguG4Llxs"
        f"/export?gid={dataset_to_gid[dataset]}&format=csv"
    )

    labels = pd.read_csv(labels_url)[cols]
    for c in cols:
        if c == "case":
            continue
        labels = labels[labels[c].notnull()]
        labels = labels[labels[c] != "None"]

    return labels.reset_index(drop=True)
