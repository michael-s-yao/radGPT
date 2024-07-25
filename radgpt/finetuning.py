
import json
import numpy as np
from typing import Dict, Sequence, Optional, Tuple

from .acr import AppropriatenessCriteria
from .data import load_case_labels, hashme, read_synthetic_dataset
from .llm import get_system_prompt


def build_finetuning_dataset(
    eval_method: str = "topic",
    val_frac: float = 0.1,
    seed: Optional[int] = 42
) -> Tuple[Sequence[Dict[str, Sequence[Dict[str, str]]]]]:
    """
    Builds the training and validation dataset for LLM fine-tuning.
    Input:
        eval_method: what evaluation metric to fine-tune the LLM on.
        val_frac: fraction of the dataset to use for model validation.
        seed: optional random seed.
    Returns:
        train: dataset for model training for fine-tuning.
        val: dataset for model validation for fine-tuning.
    """
    rng = np.random.default_rng(seed=seed)

    ac = AppropriatenessCriteria()
    y_gt = load_case_labels(dataset="synthetic")
    users = filter(
        lambda case: hashme(case) in y_gt["case"].values.tolist(),
        read_synthetic_dataset()
    )
    users = list(set(list(users)))
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
