"""
Interface for prompting publicly available LLM models.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from typing import Sequence

from .anthropic import ClaudeSonnet
from .base import LLM, get_top_k_panels
from .dbrx import DBRXInstruct
from .mistral import Mistral8x7BInstruct
from .meta import Llama3Instruct
from .openai import GPT4Turbo


__all__ = [
    "ClaudeSonnet",
    "DBRXInstruct",
    "GPT4Turbo",
    "Llama3Instruct",
    "Mistral8x7BInstruct",
    "LLM",
    "get_llm_options",
    "get_top_k_panels"
]


def get_llm_options() -> Sequence[str]:
    """
    Returns a list of all of the available LLMs.
    Input:
        None.
    Returns:
        A list of all of the available LLMs.
    """
    import sys
    from inspect import isclass
    module = sys.modules[__name__]
    opts = filter(lambda attr: isclass(getattr(module, attr)), dir(module))
    opts = filter(lambda attr: issubclass(getattr(module, attr), LLM), opts)
    return list(filter(lambda attr: getattr(module, attr) != LLM, opts))


DEFAULT_SYSTEM_PROMPT: str = (
    "You are a clinical decision support tool that classifies patient "
    "one-liners into categories. Classify each query into one of the "
    "following categories. Provide your output in JSON format with the single "
    'key "answer"\n\nCategories: {0}\n\nExample: 49M with HTN, IDDM, HLD, and '
    "20 pack-year smoking hx p/w 4 mo hx SOB and non-productive cough.\n"
    'Answer: {{"answer": "{1}"}}'
)
