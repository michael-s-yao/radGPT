"""
Mistral AI 8x7B Instruct LLM model.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import os
from mistral_inference.model import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import (
    UserMessage, SystemMessage
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from typing import Sequence

from .base import LLM


class Mistral8x7BInstruct(LLM):
    top_p: float = 0.95

    top_k: int = 50

    repetition_penalty: float = 1.01

    max_new_tokens: int = 128

    def __init__(self, seed: int = 42, **kwargs):
        """
        Args:
            seed: random seed. Default 42.
        """
        super(Mistral8x7BInstruct, self).__init__(seed=seed, **kwargs)

        self.M8x7B_DIR = os.path.join(
            os.environ["MISTRAL_MODEL"], "8x7b_instruct"
        )
        self.tokenizer = MistralTokenizer.from_file(
            os.path.join(self.M8x7B_DIR, "tokenizer.model")
        )
        self.model = Transformer.from_folder(self.M8x7B_DIR)

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
        request = ChatCompletionRequest(messages=messages)
        enc, _ = generate(
            [self.tokenizer.encode_chat_completion(request).tokens],
            self.model,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        )
        return self.tokenizer.instruct_tokenizer.tokenizer.decode(
            next(iter(enc))
        )
