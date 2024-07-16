"""
Interface for working with datasets of patient one-liner case descriptions.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import pandas as pd
from datasets import load_dataset
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
    if fn_or_url:
        return pd.read_csv(fn_or_url)

    data_files = {
        "synthetic": "synthetic.jsonl",
        "medbullets": "usmle.jsonl",
        "jama_cc": "jama.jsonl",
        "nejm": "nejm.jsonl",
        "mimic_iv": "bidmc.jsonl"
    }
    ds = load_dataset("michaelsyao/RadCases", data_files=data_files)

    labels = ds[dataset].to_pandas()
    for c in labels.columns:
        if c == "case":
            continue
        labels = labels[labels[c].notnull()]
        labels = labels[labels[c] != "None"]

    return labels.reset_index(drop=True)
