"""
BioMedGPT LLM model.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Citation(s):
    [1] Zhang K, Zhou R, Adhikarla E, et al. A generalist vision-language
        foundation model for diverse biomedical tasks. Nat Med. (2024). doi:
        https://doi.org/10.1038/s41591-024-03185-2

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Sequence

from ..utils import import_flash_attn
from .base import LLM


class BioMedGPT(LLM):
    hf_repo_name: str = "PharMolix/BioMedGPT-LM-7B"

    token: bool = True

    trust_remote_code: bool = True

    def __init__(self, seed: int = 42, **kwargs):
        """
        Args:
            seed: random seed. Default 42.
        """
        super(BioMedGPT, self).__init__(seed=seed, **kwargs)

        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16

        attn_and_autocast = import_flash_attn()
        self.attn_implementation = attn_and_autocast["attn_implementation"]
        self.autocast_context = attn_and_autocast["autocast_context"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_repo_name,
            trust_remote_code=self.trust_remote_code,
            token=self.token
        )

        chat_template_fn = os.path.join(
            os.path.dirname(__file__), "llama2_chat_template.txt"
        )
        with open(chat_template_fn, "r") as f:
            chat_template = "".join([
                x.strip().replace("\n", "") for x in f.readlines()
            ])
        # Patch from here: https://discuss.huggingface.co/t/issue-with-llama-
        # 2-chat-template-and-out-of-date-documentation/61645/3
        self.tokenizer.chat_template = chat_template

        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_repo_name,
            trust_remote_code=self.trust_remote_code,
            token=self.token,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

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

        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
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
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        raise e
                    return [str(e)]
        output = next(iter(self.tokenizer.batch_decode(enc)))
        output = output.split("[/INST]")[-1]
        if "{" in output:
            output = output[output.find("{"):]
        if "}" in output:
            output = output[:(output.rfind("}") + 1)]

        try:
            output = json.loads(output)["answer"]
            if isinstance(output, list):
                return output
            return [output]
        except (json.JSONDecodeError, KeyError, TypeError):
            return [output]
