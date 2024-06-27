"""
Anthropic AI Claude 3.5 Sonnet LLM model.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
from anthropic import AnthropicBedrock, NotGiven
from pathlib import Path
from typing import Optional, Sequence, Union

from .base import LLM


class ClaudeSonnet(LLM):
    model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0"

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
        self.client = AnthropicBedrock(aws_region="us-west-2")

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
        chat_completion = self.client.messages.create(
            system=system,
            messages=messages,
            model=self.model_name,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_new_tokens
        )

        output = chat_completion.content[0].text
        try:
            output = json.loads(output)["answer"]
            if isinstance(output, list):
                return output
            return [output]
        except json.JSONDecodeError:
            return [output]
