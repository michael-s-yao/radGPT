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
from anthropic import AnthropicBedrock
from pathlib import Path
from typing import Optional, Sequence, Union

from .base import LLM


class ClaudeSonnet(LLM):
    model_name: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    top_p: float = 0.95

    repetition_penalty: float = 1.01

    max_new_tokens: int = 128

    def __init__(
        self,
        seed: int = 42,
        aws_config_dir: Union[Path, str] = Path("~/.aws"),
        profile: Optional[str] = "default",
        **kwargs
    ):
        """
        Args:
            seed: random seed. Default 42.
            aws_config_dir: the directory path containing the AWS
                profile configuration and credentials.
            profile: the AWS profile to use.
        """
        super(ClaudeSonnet, self).__init__(seed=seed, **kwargs)

        credentials = RawConfigParser()
        credentials.read(
            os.path.expanduser(os.path.join(aws_config_dir, "credentials"))
        )
        config = RawConfigParser()
        config.read(
            os.path.expanduser(os.path.join(aws_config_dir, "config"))
        )

        self.client = AnthropicBedrock(
            aws_access_key=credentials.get(profile, "aws_access_key_id"),
            aws_secret_key=credentials.get(profile, "aws_secret_access_key"),
            aws_session_token=os.environ["AWS_SESSION_TOKEN"],
            aws_region=config.get(
                "profile " * int(profile != "default") + profile, "region"
            )
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
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        chat_completion = self.client.messages.create(
            messages=messages,
            model=self.model_name,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            frequency_penalty=self.repetition_penalty,
            response_format={"type": "json_object"},
            seed=self.seed
        )
        answer = json.loads(chat_completion.choices[0].message.content)
        answer = answer["answer"]
        if isinstance(answer, str):
            return [answer]
        return answer
