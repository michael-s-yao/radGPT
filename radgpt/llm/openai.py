"""
OpenAI GPT-4 Turbo LLM model.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import backoff
import json
import jsonlines
import os
import tempfile
from openai import OpenAI, RateLimitError
from openai.types.batch import Batch
from openai.types.fine_tuning import FineTuningJob
from pathlib import Path
from typing import Any, Dict, Sequence, Union

from .base import LLM


class GPT4Turbo(LLM):
    model_name: str = "gpt-4-turbo"

    def __init__(self, seed: int = 42, **kwargs):
        """
        Args:
            seed: random seed. Default 42.
        """
        super(GPT4Turbo, self).__init__(seed=seed, **kwargs)

        self.json_format = True
        self.client = OpenAI()

    @backoff.on_exception(backoff.expo, RateLimitError)
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
        chat_completion = self.client.chat.completions.create(
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

    def generate_batch_query(
        self, prompt: str, query_id: str
    ) -> Dict[str, Any]:
        """
        Generates a batched query request.
        Input:
            prompt: an input prompt to ask the large language model (LLM).
            query_id: the unique identifier for the query.
        Returns:
            The batched query request as an object.
        """
        messages = []
        if hasattr(self, "system_prompt") and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {
            "custom_id": query_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": messages,
                "model": self.model_name,
                "top_p": self.top_p,
                "max_tokens": self.max_new_tokens,
                "frequency_penalty": self.repetition_penalty,
                "response_format": {"type": "json_object"},
                "seed": self.seed
            }
        }

    def submit_batch_query(
        self, queries: Sequence[Dict[str, Any]], **kwargs
    ) -> Batch:
        """
        Submits a batched query request.
        Input:
            queries: a list of queries to submit.
        Returns:
            The submitted Batch object.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl")

        with open(tmp.name, "wb") as f:
            with jsonlines.Writer(f) as writer:
                writer.write_all(queries)
        with open(tmp.name, "rb") as f:
            batch_input_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        return self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=kwargs
        )

    def submit_finetuning_job(
        self,
        training_file: Union[Path, str],
        validation_file: Union[Path, str],
    ) -> FineTuningJob:
        """
        Creates and submits a model finetuning job using the specified
        training and validation datasets.
        Input:
            training_file: an OpenAI file ID or local path to a training
                dataset file.
            validation_file: an OpenAI file ID or local path to a
                validation dataset file.
        Returns:
            The submitted fine-tuning job.
        """
        if os.path.isfile(training_file):
            with open(training_file, "rb") as f:
                training_file = self.client.files.create(
                    file=f, purpose="fine-tune"
                )
        else:
            training_file = self.client.files.retrieve(training_file)

        if os.path.isfile(validation_file):
            with open(validation_file, "rb") as f:
                validation_file = self.client.files.create(
                    file=f, purpose="fine-tune"
                )
        else:
            validation_file = self.client.files.retrieve(validation_file)

        return self.client.fine_tuning.jobs.create(
            training_file=training_file.id,
            validation_file=validation_file.id,
            model=self.model_name,
            seed=self.seed
        )
