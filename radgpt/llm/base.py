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


class LLM(abc.ABC):
    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        seed_everything(self.seed)
        self.system_prompt = kwargs.get("system_prompt", None)

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
) -> Union[Sequence[str], Dict[str, Any]]:
    if method == "prompting":
        prompt = (
            f"Patient Case: {case}\n\n"
            "Which category best describes the patient's chief complaint?"
        )
        if batch_if_available and hasattr(llm, "generate_batch_query"):
            return llm.generate_batch_query(prompt, uid)
        return llm.query(prompt)
