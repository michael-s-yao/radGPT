"""
Anthropic AI Claude 3.5 Sonnet LLM model.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import os
from configparser import RawConfigParser
from anthropic import AnthropicBedrock, NotGiven
from pathlib import Path
from typing import Sequence, Union

from .base import LLM


class ClaudeSonnet(LLM):
    model_name: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    def __init__(
        self,
        seed: int = 42,
        credentials_fn: Union[Path, str] = os.path.expanduser(
            "~/.aws/credentials"
        ),
        profile: str = "default",
        **kwargs
    ):
        """
        Args:
            seed: random seed. Default 42.
            credentials_fn: the filename to the AWS credentials.
            profile: the credentials profile to use. Default `default`.
        """
        super(ClaudeSonnet, self).__init__(seed=seed, **kwargs)
        config = RawConfigParser()
        config.read(credentials_fn)
        self.client = AnthropicBedrock(
            aws_access_key=config.get(profile, "aws_access_key_id"),
            aws_secret_key=config.get(profile, "aws_secret_access_key"),
            aws_region="us-east-1"
        )

    def query(self, prompt: str) -> Sequence[str]:
        """
        Input:
            prompt: an input prompt to ask the large language model (LLM).
        Returns:
            The model response.
        """
        messages, system = [], NotGiven()
        if hasattr(self, "system_prompt") and self.system_prompt:
            system = self.system_prompt
        messages.append({"role": "user", "content": prompt})
        if self.json_format:
            messages.append({
                "role": "assistant",
                "content": "Here is the JSON requested:\n{"
            })
        chat_completion = self.client.messages.create(
            system=system,
            messages=messages,
            model=self.model_name,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_new_tokens
        )

        output = chat_completion.content[0].text
        if self.json_format:
            output = "{" + output[:(output.rfind("}") + 1)]
        try:
            output = json.loads(output)["answer"]
            if isinstance(output, list):
                return output
            return [output]
        except json.JSONDecodeError:
            return [output]
