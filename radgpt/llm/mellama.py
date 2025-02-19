"""
Me-LLaMA LLM model.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Citation(s):
    [1] Xie Q, Chen Q, Chen A, et al. Me LLaMA: Foundation large language
        models for medical applications. arXiv Preprint. (2024). doi:
        https://doi.org/10.48550/arXiv.2402.12749

The model weights can be downloaded from the original authors at
https://www.physionet.org/content/me-llama/1.0.0/MeLLaMA-70B/#files-panel

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import Sequence, Union

from ..utils import import_flash_attn
from .base import LLM


class MeLLaMA(LLM):
    token: bool = True

    max_new_tokens: int = 1024

    json_format: bool = True

    def __init__(
        self,
        modeldir: Union[Path, str] = "./MeLLaMA-70B",
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            modeldir: path to the model weights directory.
            seed: random seed. Default 42.
        """
        super(MeLLaMA, self).__init__(seed=seed, **kwargs)
        self.modeldir = modeldir

        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16

        attn_and_autocast = import_flash_attn()
        self.attn_implementation = attn_and_autocast["attn_implementation"]
        self.autocast_context = attn_and_autocast["autocast_context"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.modeldir)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.modeldir,
            device_map="auto",
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype,
        )

    def query(self, prompt: str) -> Sequence[str]:
        """
        Input:
            prompt: an input prompt to ask the large language model (LLM).
        Returns:
            The model response.
        """
        messages = []
        if hasattr(self, "system_prompt") and self.system_prompt:
            messages.append(self.system_prompt)
        messages.append(prompt)

        message = "\n".join(messages)

        tokens = self.tokenizer(message, return_tensors="pt").input_ids
        tokens = tokens.to(self.model.device)
        with torch.inference_mode():
            with self.autocast_context:
                try:
                    enc = self.model.generate(
                        input_ids=tokens,
                        max_new_tokens=self.max_new_tokens,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        use_cache=True,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        raise e
                    return [str(e)]

        output = self.tokenizer.decode(enc[0], skip_special_tokens=True)

        if self.json_format:
            output = output[:(output.rfind("}") + 1)]
            output = output[max(0, output.rfind("{")):]
            if not output.startswith("{"):
                output = "{" + output

        try:
            output = json.loads(output)["answer"]
            if isinstance(output, list):
                return output
            return [output]
        except (json.JSONDecodeError, KeyError, TypeError):
            return [output]
