"""
Meta Llama-3 70B Instruct LLM model.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import torch
from transformers import pipeline
from typing import Sequence

from ..utils import import_flash_attn
from .base import LLM


class Llama3Instruct(LLM):
    hf_repo_name: str = "meta-llama/Meta-Llama-3-70B-Instruct"

    token: bool = True

    trust_remote_code: bool = True

    def __init__(self, seed: int = 42, **kwargs):
        """
        Args:
            seed: random seed. Default 42.
        """
        super(Llama3Instruct, self).__init__(seed=seed, **kwargs)

        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16

        attn_and_autocast = import_flash_attn()
        self.autocast_context = attn_and_autocast["autocast_context"]

        self.pipeline = pipeline(
            "text-generation",
            model=self.hf_repo_name,
            model_kwargs={"torch_dtype": self.dtype},
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
            token=self.token
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

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
        with torch.inference_mode():
            with self.autocast_context:
                try:
                    output = self.pipeline(
                        messages,
                        max_new_tokens=self.max_new_tokens,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        use_cache=True,
                        do_sample=True,
                        eos_token_id=self.terminators,
                        pad_token_id=self.pipeline.tokenizer.eos_token_id,
                    )
                except RuntimeError as e:
                    return [str(e)]
        output = output[0]["generated_text"][-1]["content"]
        try:
            output = json.loads(output)["answer"]
            if isinstance(output, list):
                return output
            return [output]
        except (json.JSONDecodeError, KeyError):
            return [output]
