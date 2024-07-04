"""
Interface for prompting publicly available LLM models.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import sys
from typing import Dict, Sequence
from inspect import isclass

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
    "get_top_k_panels",
    "get_cot_reasoning_method_options",
    "get_system_prompt"
]


def get_llm_options() -> Sequence[str]:
    """
    Returns a list of all of the available LLMs.
    Input:
        None.
    Returns:
        A list of all of the available LLMs.
    """
    module = sys.modules[__name__]
    opts = filter(lambda attr: isclass(getattr(module, attr)), dir(module))
    opts = filter(lambda attr: issubclass(getattr(module, attr), LLM), opts)
    return list(filter(lambda attr: getattr(module, attr) != LLM, opts))


def get_method_options() -> Sequence[str]:
    """
    Returns the implemented LLM prompting options. `prompting` is just
    standard prompting, `rag` is retrieval-augmented generation, `icl` is
    in-context learning, and `cot` is chain-of-thought prompting.
    Input:
        None.
    Returns:
        A list of the implemented LLM prompting options.
    """
    return ["prompting", "rag", "icl", "cot"]


DEFAULT_SYSTEM_PROMPT: str = (
    "You are a clinical decision support tool that classifies patient "
    "one-liners into categories. Classify each query into one of the "
    "following categories. Provide your output in JSON format with the single "
    'key "answer"\n\nCategories: {0}\n\nExample: 49M with HTN, IDDM, HLD, and '
    "20 pack-year smoking hx p/w 4 mo hx SOB and non-productive cough.\n"
    'Answer: {{"answer": "{1}"}}'
)


RAG_SYSTEM_PROMPT: str = (
    "You are a clinical decision support tool that classifies patient "
    "one-liners into categories. Classify each query into one of the "
    "following categories. You will be given context that might be helpful, "
    "but you can ignore the context if it is not helpful. Provide your output "
    'in JSON format with the single key "answer"\n\nCategories: {0}\n\n'
    "Example: 49M with HTN, IDDM, HLD, and 20 pack-year smoking hx p/w 4 mo "
    'hx SOB and non-productive cough.\nAnswer: {{"answer": "{1}"}}'
)


COT_SYSTEM_PROMPT_BASE: str = (
    "You are a clinical decision support tool that classifies patient "
    "one-liners into categories. Classify each query into one of the "
    "following categories. Provide your output in JSON format "
    '{{"answer": [YOUR CLASSIFICATION], "rationale": [YOUR STEP-BY-STEP '
    "REASONING]}}\n\nCategories: {0}\n\nUse step-by-step deduction to "
    "identify the correct classification."
)


COT_SYSTEM_PROMPT_ANALYTIC: str = (
    "You are a clinical decision support tool that classifies patient "
    "one-liners into categories. Classify each query into one of the "
    "following categories. Provide your output in JSON format "
    '{{"answer": [YOUR CLASSIFICATION], "rationale": [YOUR STEP-BY-STEP '
    "REASONING]}}\n\nCategories: {0}\n\nUse analytic reasoning to deduce the "
    "physiologic or biochemical pathophysiology of the patient and step by "
    "step identify the correct response."
)


COT_SYSTEM_PROMPT_DIFFERENTIAL_DIAGNOSIS: str = (
    "You are a clinical decision support tool that classifies patient "
    "one-liners into categories. Classify each query into one of the "
    "following categories. Provide your output in JSON format "
    '{{"answer": [YOUR CLASSIFICATION], "rationale": [YOUR STEP-BY-STEP '
    "REASONING]}}\n\nCategories: {0}\n\nUse step-by-step deduction to "
    "first create a differential diagnosis, and select the answer that is "
    "most consistent with your reasoning."
)


COT_SYSTEM_PROMPT_BAYESIAN: str = (
    "You are a clinical decision support tool that classifies patient "
    "one-liners into categories. Classify each query into one of the "
    "following categories. Provide your output in JSON format "
    '{{"answer": [YOUR CLASSIFICATION], "rationale": [YOUR STEP-BY-STEP '
    "REASONING]}}\n\nCategories: {0}\n\nIn your rationale, use step-by-step "
    "Bayesian Inference to create a prior probability that is updated with "
    "new information in the history to produce a posterior probability and "
    "determine the final classification."
)


COT_REASONING_METHOD_SYSTEM_PROMPTS: Dict[str, str] = {
    "default": COT_SYSTEM_PROMPT_BASE,
    "analytic": COT_SYSTEM_PROMPT_ANALYTIC,
    "bayesian": COT_SYSTEM_PROMPT_BAYESIAN,
    "differential": COT_SYSTEM_PROMPT_DIFFERENTIAL_DIAGNOSIS
}


def get_cot_reasoning_method_options() -> Sequence[str]:
    """
    Returns the implemented chain-of-thought reasoning methods.
    Input:
        None.
    Returns:
        A list of the implemented chain-of-thought reasoning methods.
    """
    return list(COT_REASONING_METHOD_SYSTEM_PROMPTS.keys())


def get_system_prompt(method: str, **kwargs) -> str:
    """
    Returns the system prompt to use for the LLM prompting method.
    Input:
        method: the LLM prompting method.
    Returns:
        The corresponding system prompt.
    """
    if method.lower() in ["prompting", "icl"]:
        return DEFAULT_SYSTEM_PROMPT
    elif method.lower() == "rag":
        return RAG_SYSTEM_PROMPT
    elif method.lower() == "cot":
        assert "rationale" in kwargs.keys(), (
            "Chain-of-thought reasoning method must be specified."
        )
        return COT_REASONING_METHOD_SYSTEM_PROMPTS[
            kwargs["rationale"].lower()
        ]
    elif method.lower() == "icl":
        return DEFAULT_SYSTEM_PROMPT
    raise NotImplementedError
