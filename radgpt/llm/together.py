"""
Together AI-hosted LLM models.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import os
from pathlib import Path
from pydantic import BaseModel, Field
from together import Together
from typing import Sequence, Union

from .base import LLM


class TogetherAILLM(LLM):
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            model_name: the name of the model to use.
            seed: random seed. Default 42.
        """
        super(TogetherAILLM, self).__init__(seed=seed, **kwargs)
        self.model = model_name
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

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

        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_new_tokens
        )

        response = chat_completion.choices[0].message.content
        if self.json_format:
            response = response[:(response.rfind("}") + 1)]
            if not response.startswith("{"):
                response = "{" + response

        try:
            output = json.loads(response)["answer"]
            if isinstance(output, list):
                return output
            return [output]
        except (json.JSONDecodeError, KeyError):
            return [response]
