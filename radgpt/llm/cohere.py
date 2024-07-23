"""
Cohere C4AI Command R LLM model.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Sequence

from ..utils import import_flash_attn
from .base import LLM


class CommandRPlus(LLM):
    hf_repo_name: str = "CohereForAI/c4ai-command-r-plus-4bit"

    def __init__(self, seed: int = 42, **kwargs):
        """
        Args:
            seed: random seed. Default 42.
        """
        super(CommandRPlus, self).__init__(seed=seed, **kwargs)

        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16

        attn_and_autocast = import_flash_attn()
        self.attn_implementation = attn_and_autocast["attn_implementation"]
        self.autocast_context = attn_and_autocast["autocast_context"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_repo_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_repo_name,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        self.start_of_turn = "<|START_OF_TURN_TOKEN|>"
        self.end_of_turn = "<|END_OF_TURN_TOKEN|>"
        self.chatbot = "<|CHATBOT_TOKEN|>"

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
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        raise e
                    return [str(e)]
        output = self.tokenizer.decode(enc[0])
        output = output.split(self.start_of_turn)[-1]
        output = output.split(self.end_of_turn)[0]
        output = output.split(self.chatbot)[-1]

        if self.json_format:
            output = output[:(output.rfind("}") + 1)]
            if not output.startswith("{"):
                output = "{" + output

        try:
            output = json.loads(output)["answer"]
            if isinstance(output, list):
                return output
            return [output]
        except json.JSONDecodeError:
            return [output]
