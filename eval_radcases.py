#!/usr/bin/env python3
"""
Semantic quality evaluation of the RadCases Dataset.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import os
import jsonlines
import numpy as np
import pandas as pd
import torch
import tiktoken
from datasets import load_dataset
from evaluate import load
from pathlib import Path
from typing import Optional, Sequence, Union

import radgpt


def compute_perplexity(
    dataset_name: str,
    cases: Sequence[str],
    savedir: Optional[Union[Path, str]] = None,
    model: str = "Locutusque/gpt2-large-medical",
    max_length: Optional[int] = 1024
) -> np.ndarray:
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(
        predictions=cases, model_id=model, max_length=max_length
    )
    scores = np.array(results["perplexities"])
    if savedir is not None:
        try:
            os.makedirs(savedir, exist_ok=True)
        except FileNotFoundError:
            pass
        dataset_name = dataset_name.split("/")[-1]
        np.save(os.path.join(savedir, f"{dataset_name}.npy"), scores)
    return scores


def compute_token_counts(
    dataset_name: str,
    cases: Sequence[str],
    savedir: Optional[Union[Path, str]] = None,
    model: str = "gpt-4o"
) -> np.ndarray:
    encoder = tiktoken.encoding_for_model(model)
    tok_cts = np.array([len(encoder.encode(c)) for c in cases])
    if savedir is not None:
        try:
            os.makedirs(savedir, exist_ok=True)
        except FileNotFoundError:
            pass
        dataset_name = dataset_name.split("/")[-1]
        np.save(os.path.join(savedir, f"{dataset_name}.npy"), tok_cts)
    return tok_cts


def get_random_bidmc(
    notes_fn: Union[Path, str],
    n: int,
    full_notes: bool = False,
    seed: Optional[int] = None
) -> Sequence[str]:
    """
    Retrieves a random set of n input one-liners.
    Input:
        notes_fn: a file path to a CSV dataset of notes from MIMIC-IV.
        n: the maximum number of notes to retrieve.
        full_notes: whether to return whole notes or singular sentences.
        seed: optional random seed.
    Returns:
        A list of the input one-liner strings.
    """
    data = pd.read_csv(notes_fn)["text"].to_numpy()
    data = np.array([note.strip().replace("\n", "") for note in data])
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    if n <= 0:
        n = len(data)
    data = data[:min(n, len(data))]
    if full_notes:
        return data
    for i in range(len(data)):
        data[i] = rng.choice(
            list(
                filter(
                    lambda stn: len(stn) > 5,
                    radgpt.utils.split_into_sentences(data[i])
                )
            )
        )
    return data


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=str,
    required=True,
    help="The dataset of patient one-liners to use."
)
@click.option(
    "-m",
    "--metric",
    type=click.Choice(["similarity", "tok", "perplexity"]),
    required=True,
    help="NLP metric to use."
)
@click.option(
    "-c",
    "--cache-fn",
    type=str,
    default="docs/gt.pt",
    show_default=True,
    help="The optional cache file for the reference corpus embeddings."
)
@click.option(
    "--max-cases",
    type=int,
    default=10000,
    show_default=True,
    help="The maximum number of cases to use."
)
@click.option(
    "--seed", type=int, default=42, show_default=True, help="Random seed."
)
def main(
    dataset: str,
    metric: str,
    cache_fn: Optional[Union[Path, str]] = None,
    max_cases: Optional[int] = -1,
    seed: Optional[int] = None
):
    """Semantic quality evaluation of the RadCases dataset."""
    # Load the specified dataset of patient one-liners.
    if dataset.lower() in [
        x.lower() for x in radgpt.utils.get_experiment_options()
    ]:
        y_gt = radgpt.data.load_case_labels(dataset=dataset)
        patient_cases = filter(
            lambda case: (
                radgpt.data.hashme(case) in y_gt["case"].values.tolist()
            ),
            getattr(radgpt.data, f"read_{dataset}_dataset")()
        )
        patient_cases = sorted(
            list(set(list(patient_cases))), key=radgpt.data.hashme
        )
    elif dataset.lower() == "gt":
        with open("radGPT-UI/app/static/assets/cases.jsonl", "r") as f:
            with jsonlines.Reader(f) as reader:
                patient_cases = sorted(list(set([x["case"] for x in reader])))
    elif dataset.lower() == "bidmc_random":
        patient_cases = get_random_bidmc(
            "radgpt/data/discharge.csv.gz", -1, seed=seed
        )
    elif dataset.lower() == "bidmc_full":
        patient_cases = get_random_bidmc(
            "radgpt/data/discharge.csv.gz", -1, full_notes=True, seed=seed
        )
    elif dataset.lower() == "bidmc_rad":
        patient_cases = get_random_bidmc("radiology.csv.gz", 128, seed=seed)
    else:
        ds_kwargs = {
            "split": "train", "trust_remote_code": True, "path": dataset
        }
        if dataset.lower() == "wikitext":
            ds_kwargs.update({"name": "wikitext-2-raw-v1"})
        patient_cases = load_dataset(**ds_kwargs)
        patient_cases.shuffle(seed=seed)
        cases = []
        for row in patient_cases:
            if dataset.lower() == "pubmed":
                row = row["MedlineCitation"]["Article"]
                row = row["Abstract"]["AbstractText"]
            elif dataset.lower() == "wikitext":
                row = row["text"]
            elif dataset.lower() == "bigbio/med_qa":
                row = row["question"]
            elif dataset.lower() == "maartengr/arxiv_nlp":
                row = row["Abstracts"].replace("\n", " ").strip()

            if len(row) == 0:
                continue
            cases.append(row)
            if max_cases > 0 and len(cases) >= max_cases:
                break
        patient_cases = cases

    if metric.lower() == "tok":
        compute_token_counts(dataset, patient_cases, metric.lower())
        return
    elif metric.lower() == "perplexity":
        compute_perplexity(dataset, patient_cases, metric.lower())
        return

    model = radgpt.utils.NVEmbedv2()

    # Load the ground truth reference of true patient one-liners.
    if cache_fn is not None and os.path.isfile(cache_fn):
        corpus = torch.load(cache_fn).to(
            device=next(model.parameters()).device, dtype=model.dtype
        )
    else:
        with open("radGPT-UI/app/static/assets/cases.jsonl", "r") as f:
            with jsonlines.Reader(f) as reader:
                cases = sorted(list(set([x["case"] for x in reader])))
        corpus = model(cases)
        if cache_fn is not None:
            try:
                os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
            except FileNotFoundError:
                pass
            torch.save(corpus.detach().cpu(), cache_fn)

    scores = torch.vstack([
        model.similarity_score(case, corpus).detach().cpu()
        for case in patient_cases
    ])
    os.makedirs(metric.lower(), exist_ok=True)
    dataset = dataset.split("/")[-1]
    torch.save(scores, os.path.join(metric.lower(), f"{dataset}_simscores.pt"))


if __name__ == "__main__":
    main()
