#!/usr/bin/env python3
"""
Finetune LLM models on the synthetic RadCases dataset.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import jsonlines
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
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
    "--llm",
    type=click.Choice(
        radgpt.llm.get_llm_options(), case_sensitive=False
    ),
    default="GPT4Turbo",
    show_default=True,
    help="The base LLM model to use."
)
@click.option(
    "--model-name",
    type=str,
    default="gpt-4o-mini-2024-07-18",
    show_default=True,
    help="The LLM model name to use."
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
    train_fn: Optional[Union[Path, str]] = None,
    val_fn: Optional[Union[Path, str]] = None,
    llm: str = "GPT4Turbo",
    model_name: Optional[str] = "gpt-4o-mini-2024-07-18",
    eval_method: str = "topic",
    val_frac: float = 0.1,
    seed: int = 42
):
    """Finetune an LLM model on the synthetic RadCases dataset."""
    if train_fn is None or val_fn is None:
        train, val = radgpt.finetuning.build_finetuning_dataset(
            eval_method=eval_method, val_frac=val_frac, seed=seed
        )
        train_tmp, val_tmp = NamedTemporaryFile(), NamedTemporaryFile()
        train_fn, val_fn = train_tmp.name, val_tmp.name
        write_jsonlines_to_tmp(train, train_fn)
        write_jsonlines_to_tmp(val, val_fn)

    llm = getattr(radgpt.llm, llm)(seed=seed)
    if model_name is not None and model_name.title() != "None":
        llm.model_name = model_name
    click.echo(llm.submit_finetuning_job(train_fn, val_fn))


if __name__ == "__main__":
    main()
