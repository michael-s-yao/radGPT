"""
Meta Llama-3 70B Instruct LLM model.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import torch
from datasets import Dataset, load_dataset
from pathlib import Path
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import SFTTrainer, SFTConfig
from typing import Any, Dict, Optional, Sequence, Union

from ..utils import import_flash_attn
from .base import LLM


class Llama3Instruct(LLM):
    hf_repo_name: str = "meta-llama/Meta-Llama-3-70B-Instruct"

    token: bool = True

    trust_remote_code: bool = True

    LLAMA_3_CHAT_FINETUNING_TEMPLATE: str = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '\n\nHuman: ' + message['content'] + eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '\n\nAssistant: ' + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '\n\nAssistant: ' }}"
        "{% endif %}"
    )

    def __init__(self, seed: int = 42, **kwargs):
        """
        Args:
            seed: random seed. Default 42.
        """
        super(Llama3Instruct, self).__init__(seed=seed, **kwargs)

        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16

        attn_and_autocast = import_flash_attn()
        self.autocast_context = attn_and_autocast["autocast_context"]

        self.pipeline = pipeline(
            "text-generation",
            model=self.hf_repo_name,
            model_kwargs={"torch_dtype": self.dtype},
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
            token=self.token
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

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

        with torch.inference_mode():
            with self.autocast_context:
                try:
                    output = self.pipeline(
                        messages,
                        max_new_tokens=self.max_new_tokens,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        use_cache=True,
                        do_sample=True,
                        eos_token_id=self.terminators,
                        pad_token_id=self.pipeline.tokenizer.eos_token_id,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        raise e
                    return [str(e)]
        output = output[0]["generated_text"][-1]["content"]

        if self.json_format:
            output = output[:(output.rfind("}") + 1)]
            if not output.startswith("{"):
                output = "{" + output

        try:
            output = json.loads(output)["answer"]
            if isinstance(output, list):
                return output
            return [output]
        except (json.JSONDecodeError, KeyError):
            return [output]

    @classmethod
    def submit_finetuning_job(
        cls,
        args: SFTConfig,
        train: Union[Path, str, Dataset],
        val: Union[Path, str, Dataset],
        fast_dev_run: bool = False,
        peft_config_kwargs: Optional[Dict[str, Any]] = {},
        **kwargs
    ) -> None:
        """
        Creates and submits a model finetuning job using the specified
        training and validation datasets.
        Input:
            train: a file ID or local path to a training dataset.
            val: a file ID or local path to a validation dataset.
        Returns:
            None.
        """
        if not isinstance(train, Dataset):
            train = load_dataset(train)
        if not isinstance(val, Dataset):
            val = load_dataset(val)

        tokenizer = AutoTokenizer.from_pretrained(
            cls.hf_repo_name, use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.chat_template = cls.LLAMA_3_CHAT_FINETUNING_TEMPLATE

        def template_dataset(examples):
            return {
                "text": tokenizer.apply_chat_template(
                    examples["messages"], tokenize=False
                )
            }

        train = train.map(template_dataset, remove_columns=["messages"])
        val = val.map(template_dataset, remove_columns=["messages"])

        peft_config_default_kwargs = {
            "lora_alpha": 8,
            "lora_dropout": 0.05,
            "r": 16,
            "bias": "none",
            "target_modules": "all-linear",
            "task_type": "CAUSAL_LM",
        }
        peft_config_default_kwargs.update(peft_config_kwargs)
        peft_config = LoraConfig(peft_config_default_kwargs)

        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

        attn_and_autocast = import_flash_attn()
        autocast_context = attn_and_autocast["autocast_context"]

        with autocast_context:
            model = AutoModelForCausalLM.from_pretrained(
                cls.hf_repo_name,
                attn_implementation="sdpa",
                torch_dtype=dtype,
                use_cache=(not args.gradient_checkpointing)
            )

            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()

            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train,
                eval_dataset=val,
                peft_config=peft_config,
                tokenizer=tokenizer,
            )
            if trainer.accelerator.is_main_process:
                trainer.model.print_trainable_parameters()

            trainer.train(
                resume_from_checkpoint=args.resume_from_checkpoint
            )

        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
                "FULL_STATE_DICT"
            )
        return trainer.save_model()


class FineTunedLlama3Instruct(LLM):
    def __init__(self, model_name: Union[Path, str], seed: int = 42, **kwargs):
        """
        Args:
            model_name: the ID of the fine-tuned model.
            seed: random seed. Default 42.
        """
        super(FineTunedLlama3Instruct, self).__init__(seed=seed, **kwargs)

        self.model_name = model_name
        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16

        attn_and_autocast = import_flash_attn()
        self.autocast_context = attn_and_autocast["autocast_context"]
        self.model = AutoPeftModelForCausalLM.from_pretrained(
          self.model_name,
          torch_dtype=self.dtype,
          quantization_config={"load_in_4bit": True},
          device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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

        with torch.inference_mode():
            with self.autocast_context:
                tokens = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )

                outputs = self.model.generate(
                    tokens.to(self.model.device),
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty
                )
        response = self.tokenizer.decode(
            outputs[0][tokens.shape[-1]:], skip_special_tokens=True
        )

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
