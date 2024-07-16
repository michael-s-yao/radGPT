"""
Mistral AI 8x7B Instruct LLM model.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import torch
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import (
    AssistantMessage, SystemMessage, UserMessage
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import AutoModelForCausalLM
from typing import Sequence

from ..utils import import_flash_attn
from .base import LLM


class Mistral8x7BInstruct(LLM):
    hf_repo_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    token: bool = True

    trust_remote_code: bool = True

    def __init__(self, seed: int = 42, **kwargs):
        """
        Args:
            seed: random seed. Default 42.
        """
        super(Mistral8x7BInstruct, self).__init__(seed=seed, **kwargs)

        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16

        attn_and_autocast = import_flash_attn()
        self.attn_implementation = attn_and_autocast["attn_implementation"]
        self.autocast_context = attn_and_autocast["autocast_context"]

        self.tokenizer = MistralTokenizer.v1()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_repo_name,
            trust_remote_code=self.trust_remote_code,
            token=self.token,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype,
            device_map="auto",
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
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(UserMessage(content=prompt))

        if self.json_format:
            messages.append(
                AssistantMessage(content="Here is the JSON requested:\n{")
            )

        request = ChatCompletionRequest(messages=messages)
        tokens = self.tokenizer.encode_chat_completion(request).tokens
        eos_token_id = self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        with torch.inference_mode():
            with self.autocast_context:
                enc = self.model.generate(
                    torch.tensor([tokens]).to(self.model.device),
                    max_new_tokens=self.max_new_tokens,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    use_cache=True,
                    do_sample=True,
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id
                )
        output = self.tokenizer.decode(enc[0].tolist())
        output = output.split(" [/INST] ", 1)[-1].split("}", 1)[0] + "}"

        if self.json_format:
            output = "{" + output[:(output.rfind("}") + 1)]

        try:
            output = json.loads(output)["answer"]
            if isinstance(output, list):
                return output
            return [output]
        except json.JSONDecodeError:
            return [output]
