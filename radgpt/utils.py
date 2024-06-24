"""
Utility functions for running LLM model inference and additional experiments.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import torch
from contextlib import nullcontext
from typing import ContextManager, Dict, Sequence, Union

from .data import __all__ as data_fns


__all__ = [
    "import_flash_attn",
    "get_experiment_options"
]


def import_flash_attn() -> Dict[str, Union[str, ContextManager]]:
    """
    Attempts to import and use FlashAttention for LLM model inference.
    Input:
        None.
    Returns:
        A dictionary containing the following key-value pairs:
            attn_implementation: the attention implementation to use.
            autocast_context: the corresponding autocast context manager.
    Citation(s):
        [1] Dao T, Fu DY, Ermon S, Rudra A, Re C. FlashAttention: Fast and
            memory-efficient exact attention with IO-awarness. arxiv Preprint.
            (2023). doi: 10.48550/arXiv.2205.14135
    """
    try:
        import flash_attn  # noqa
        assert torch.cuda.is_available()
        return {
            "attn_implementation": "flash_attention_2",
            "autocast_context": torch.autocast("cuda", torch.bfloat16)
        }
    except ImportError:
        return {
            "attn_implementation": "eager", "autocast_context": nullcontext()
        }


def get_experiment_options() -> Sequence[str]:
    """
    Returns a list of the implemented experiment options to run.
    Input:
        None.
    Returns:
        A list of the implemented experiment options to run.
    """
    pref, suff = "read_", "_dataset"
    options = filter(
        lambda fn: fn.startswith(pref) and fn.endswith(suff), data_fns
    )
    options = map(
        lambda fn: (
            fn.replace(pref, "", 1)[::-1].replace(suff[::-1], "", 1)[::-1]
        ),
        list(options)
    )
    return list(options)
