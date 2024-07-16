"""
Base large language model (LLM) class definition.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import abc
import warnings
from pytorch_lightning import seed_everything
from typing import Any, Dict, Optional, Sequence, Union

from ..acr import AppropriatenessCriteria
from ..retrieval import Document


# Baseline user prompt without any engineering.
USER_PROMPT_BASE: str = (
    "Patient Case: {case}\n\n"
    "Which category best describes the patient's chief complaint?"
)


# User prompt for retrieval-augmented generation (RAG).
USER_PROMPT_RAG: str = (
    "Here is some context for you to consider:\n{context}\n\n"
    f"### User:\n{USER_PROMPT_BASE}\n\n### Assistant:\n"
)


# User prompt for in-context learning (ICL).
USER_PROMPT_ICL: str = f"{{context}}\n\n{USER_PROMPT_BASE}"


class LLM(abc.ABC):
    top_p: float = 0.95

    top_k: int = 50

    repetition_penalty: float = 1.01

    max_new_tokens: int = 128

    json_format: bool = False

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        seed_everything(self.seed)
        self.system_prompt = kwargs.get("system_prompt", None)
        for key, val in kwargs.items():
            setattr(self, key, val)

    @abc.abstractmethod
    def query(self, prompt: str) -> Sequence[str]:
        """
        Input:
            prompt: an input prompt to ask the large language model (LLM).
        Returns:
            The model response.
        """
        raise NotImplementedError

    def set_system_prompt(self, prompt: str) -> None:
        if self.system_prompt is not None:
            warnings.warn(
                "System prompt is already set. Overriding...", UserWarning
            )
        self.system_prompt = prompt


def get_top_k_panels(
    case: str,
    criteria: AppropriatenessCriteria,
    llm: LLM,
    top_k: int,
    method: str,
    uid: Optional[str] = None,
    batch_if_available: bool = True,
    rag_context: Optional[Sequence[Document]] = None,
    icl_context: Optional[Sequence[str]] = None
) -> Union[Sequence[str], Dict[str, Any]]:
    """
    Returns the top k predictions for an input patient case.
    Input:
        case: the input patient case.
        criteria: the reference ACR Appropriateness Criteria.
        llm: the Large Language Model to use to process the patient case.
        top_k: the number of predictions to return.
        method: the method to use to generate the predictions.
        uid: an optional unique ID for the query.
        batch_if_available: generate a batch API request if available.
        rag_context: an optional list of contexts to use for RAG.
        icl_context: an optional list of contexts to use for ICL.
    Returns:
        The top k predictions from the LLM, or a batch request if applicable.
    """
    if method.lower() == "prompting":
        prompt = USER_PROMPT_BASE.format(case=case)
    elif method.lower() == "rag":
        prompt = USER_PROMPT_RAG.format(
            case=case, context=("\n".join([doc.text for doc in rag_context]))
        )
    elif method.lower() == "cot":
        prompt = USER_PROMPT_BASE.format(case=case)
    elif method.lower() == "icl":
        prompt = USER_PROMPT_ICL.format(case=case, context=icl_context)
    else:
        raise NotImplementedError

    if batch_if_available and hasattr(llm, "generate_batch_query"):
        return llm.generate_batch_query(prompt, uid)
    return llm.query(prompt)
