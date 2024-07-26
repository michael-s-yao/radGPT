"""
Utility functions for model finetuning.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, Sequence, Optional, Tuple

from .acr import AppropriatenessCriteria
from .data import (
    load_case_labels,
    hashme,
    read_synthetic_dataset,
    read_medbullets_dataset,
    read_jama_cc_dataset,
    read_nejm_dataset,
    read_mimic_iv_dataset
)
from .llm import get_system_prompt
from .utils import get_experiment_options


def get_finetuning_partition_options() -> Sequence[str]:
    """
    Returns the RadCases dataset partitions implemented for finetuning.
    Input:
        None.
    Returns:
        The RadCases dataset partitions implemented for finetuning.
    """
    return ["synthetic", "mixed"]


def build_finetuning_dataset(
    partition: str = "synthetic",
    eval_method: str = "topic",
    val_frac: float = 0.1,
    seed: Optional[int] = 42
) -> Tuple[Sequence[Dict[str, Sequence[Dict[str, str]]]]]:
    """
    Builds the training and validation dataset for LLM fine-tuning.
    Input:
        partition: the partition of the RadCases dataset to use for
            fine-tuning.
        eval_method: what evaluation metric to fine-tune the LLM on.
        val_frac: fraction of the dataset to use for model validation.
        seed: optional random seed.
    Returns:
        train: dataset for model training for fine-tuning.
        val: dataset for model validation for fine-tuning.
    """
    if partition.lower() not in get_finetuning_partition_options():
        raise NotImplementedError

    rng = np.random.default_rng(seed=seed)
    ac = AppropriatenessCriteria()

    if partition.lower() == "synthetic":
        y_gt = load_case_labels(dataset="synthetic")
        opts = [read_synthetic_dataset()]
    else:
        y_gt = pd.concat([
            load_case_labels(dataset=ds).sample(
                n=50, replace=False, random_state=seed
            )
            for ds in get_experiment_options()
        ])
        opts = [
            read_synthetic_dataset(),
            read_medbullets_dataset(),
            read_jama_cc_dataset(),
            read_nejm_dataset(),
            read_mimic_iv_dataset()
        ]
    users = filter(
        lambda case: hashme(case) in y_gt["case"].values.tolist(),
        np.concatenate(opts)
    )
    users = sorted(list(set(list(users))), key=hashme)
    val_idxs = rng.choice(
        len(users),
        size=round(min(max(val_frac, 0.0), 1.0) * len(users)),
        replace=False
    )

    categories = "; ".join(
        ac.panels
        if eval_method == "panel"
        else (ac.topics if eval_method == "topic" else ac.studies)
    )
    ex_answer = "Thoracic" if eval_method == "panel" else (
        "Lung Cancer Screening"
        if eval_method == "topic"
        else "CT chest without IV contrast screening"
    )
    system = get_system_prompt("prompting").format(categories, ex_answer)
    systems = [system] * len(users)

    _k = "panel" if eval_method == "panel" else "topic"
    assistants = [
        json.dumps({"answer": y_gt[y_gt["case"] == hashme(case)][_k].item()})
        for case in users
    ]

    ds = [
        {
            "messages": [
                {"role": "system", "content": s},
                {"role": "user", "content": u},
                {"role": "assistant", "content": a}
            ]
        }
        for i, (s, u, a) in enumerate(zip(systems, users, assistants))
    ]
    train = [ds[idx] for idx in range(len(ds)) if idx not in val_idxs]
    val = [ds[idx] for idx in range(len(ds)) if idx in val_idxs]
    return train, val
