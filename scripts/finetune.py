#!/usr/bin/env python3
"""
Finetune LLM models on the synthetic RadCases dataset.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Portions of this script were adapted from the `run_fsdp_qlora.py` script from
@philschmid at https://github.com/philschmid/deep-learning-pytorch-huggingface/
blob/4a0caadaf58e6f9ac64be95250ac5a49193b308b/training/scripts/
run_fsdp_qlora.py.

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import jsonlines
import os
import pandas as pd
import sys
import yaml
from datasets import Dataset
from pathlib import Path
from tempfile import NamedTemporaryFile
from trl import SFTConfig
from typing import Dict, Optional, Sequence, Union

sys.path.append(".")  # noqa
import radgpt  # noqa


def write_jsonlines_to_tmp(
    ds: Sequence[Dict[str, Sequence[Dict[str, str]]]],
    tmp_fn: Union[Path, str]
) -> None:
    """
    Writes a jsonlines dataset to a named temporary file.
    Input:
        ds: a jsonlines dataset.
        tmp: the filepath of the named temporary file to write to.
    Returns:
        None.
    """
    with open(tmp_fn, "w") as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(ds)


@click.command()
@click.option(
    "--partition",
    type=click.Choice(radgpt.finetuning.get_finetuning_partition_options()),
    default="synthetic",
    show_default=True,
    help="The partition of the RadCases dataset to use for fine-tuning."
)
@click.option(
    "--train-fn",
    type=str,
    default=None,
    show_default=True,
    help="An optional existing file path or ID to use for training."
)
@click.option(
    "--val-fn",
    type=str,
    default=None,
    show_default=True,
    help="An optional existing file path or ID to use for validation."
)
@click.option(
    "--config",
    type=str,
    default=None,
    show_default=True,
    help="An optional training configuration for SFTConfig initialization."
)
@click.option(
    "--llm",
    type=click.Choice(
        radgpt.llm.get_llm_options(), case_sensitive=False
    ),
    default="GPT4Turbo",
    show_default=True,
    help="The base LLM model to use."
)
@click.option(
    "--by-panel",
    "eval_method",
    type=str,
    flag_value="panel",
    help="Whether to evaluate the LLM by ACR AC Panels."
)
@click.option(
    "--by-topic",
    "eval_method",
    type=str,
    flag_value="topic",
    default=True,
    show_default=True,
    help="Whether to evaluate the LLM by ACR AC Topics."
)
@click.option(
    "--by-study",
    "eval_method",
    type=str,
    flag_value="study",
    help="Whether to evaluate the LLM by imaging studies."
)
@click.option(
    "--val-frac",
    type=float,
    default=0.1,
    show_default=True,
    help="Fraction of the dataset to use for model validation."
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed."
)
def main(
    partition: str,
    train_fn: Optional[Union[Path, str]] = None,
    val_fn: Optional[Union[Path, str]] = None,
    config: Optional[Union[Path, str]] = None,
    llm: str = "GPT4oMini",
    eval_method: str = "topic",
    val_frac: float = 0.1,
    seed: int = 42
):
    """Finetune an LLM model on the synthetic RadCases dataset."""
    llm, llm_name = getattr(radgpt.llm, llm), llm

    if train_fn is None or val_fn is None:
        train, val = radgpt.finetuning.build_finetuning_dataset(
            partition=partition,
            eval_method=eval_method,
            val_frac=val_frac,
            seed=seed
        )
        if issubclass(llm, radgpt.llm.OpenAIModel):
            train_tmp, val_tmp = NamedTemporaryFile(), NamedTemporaryFile()
            train_fn, val_fn = train_tmp.name, val_tmp.name
            write_jsonlines_to_tmp(train, train_fn)
            write_jsonlines_to_tmp(val, val_fn)
            train_input, val_input = train_fn, val_fn
        else:
            train_input = Dataset.from_pandas(pd.json_normalize(train))
            val_input = Dataset.from_pandas(pd.json_normalize(val))

    if issubclass(llm, radgpt.llm.OpenAIModel):
        click.echo(
            llm(seed=seed).submit_finetuning_job(train_input, val_input)
        )
    else:
        assert config is not None and os.path.isfile(config)
        with open(config, "r") as f:
            args = SFTConfig(
                output_dir=f"./{llm_name}-{partition}", **yaml.safe_load(f)
            )

        if args.gradient_checkpointing:
            args.gradient_checkpointing_kwargs = {"use_reentrant": True}

        llm.submit_finetuning_job(args, train_input, val_input)


if __name__ == "__main__":
    main()
