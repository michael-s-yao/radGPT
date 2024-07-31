"""
Base large language model (LLM) class definition.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import abc
import json
import torch
import warnings
from datasets import Dataset
from pathlib import Path
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, Optional, Sequence, Union

from ..acr import AppropriatenessCriteria
from ..retrieval import Document
from ..utils import import_flash_attn


# Baseline user prompt without any engineering.
USER_PROMPT_BASE: str = (
    "Patient Case: {case}\n\n"
    "Which category best describes the patient's chief complaint?"
)


# Baseline user prompt without any engineering.
USER_PROMPT_BASE_STUDY: str = (
    "Patient Case: {case}\n\n"
    "Which imaging study (if any) is most appropriate for this patient?"
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
        """
        Sets the system prompt of the language model to the input prompt.
        Input:
            prompt: the system prompt for the language model.
        Returns:
            None.
        """
        if self.system_prompt is not None:
            warnings.warn(
                "System prompt is already set. Overriding...", UserWarning
            )
        self.system_prompt = prompt

    @classmethod
    def submit_finetuning_job(
        cls,
        train: Union[Path, str, Dataset],
        val: Union[Path, str, Dataset],
        **kwargs
    ) -> Optional[Any]:
        """
        Creates and submits a model finetuning job using the specified
        training and validation datasets.
        Input:
            train: a file ID or local path to a training dataset.
            val: a file ID or local path to a validation dataset.
        Returns:
            Varies by implementation.
        """
        raise NotImplementedError


class FineTunedLocalLLM(LLM):
    def __init__(self, model_name: Union[Path, str], seed: int = 42, **kwargs):
        """
        Args:
            model_name: the local model directory of the fine-tuned model.
            seed: random seed. Default 42.
        """
        super(FineTunedLocalLLM, self).__init__(seed=seed, **kwargs)

        self.model_path = model_name
        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16

        attn_and_autocast = import_flash_attn()
        self.autocast_context = attn_and_autocast["autocast_context"]
        self.model = AutoModelForCausalLM.from_pretrained(
          self.model_path,
          torch_dtype=self.dtype,
          trust_remote_code=True,
          device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    @torch.inference_mode()
    def query(self, prompt: str) -> Sequence[str]:
        """
        Input:
            prompt: an input prompt to ask the large language model (LLM).
        Returns:
            The model response.
        """
        messages = []
        if hasattr(self, "system_prompt") and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self.json_format:
            messages.append({
                "role": "assistant",
                "content": "Here is the JSON requested:\n{"
            })

        with self.autocast_context:
            tokens = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt"
            )

            outputs = self.model.generate(
                tokens.to(self.model.device),
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        sidx = max(response.find("assistant"), 0) + len("assistant")
        eidx = response.find("}", sidx) + 1
        response = response[sidx:eidx].strip()
        if not response.startswith("{"):
            response = "{" + response

        try:
            output = json.loads(response)["answer"]
            if isinstance(output, list):
                return output
            return [output]
        except (json.JSONDecodeError, KeyError):
            return [response]


def get_top_k_panels(
    case: str,
    criteria: AppropriatenessCriteria,
    llm: LLM,
    top_k: int,
    method: str,
    uid: Optional[str] = None,
    batch_if_available: bool = True,
    rag_context: Optional[Sequence[Document]] = None,
    icl_context: Optional[Sequence[str]] = None,
    **kwargs
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
    if method.lower() in ["prompting", "ft"]:
        if kwargs.get("study", False):
            prompt = USER_PROMPT_BASE_STUDY.format(case=case)
        else:
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
