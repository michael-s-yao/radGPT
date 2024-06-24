#!/usr/bin/env python3
"""
Aligning large language models (LLMs) with ACR Appropriateness Criteria.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import json
import os
from pathlib import Path
from typing import Optional, Union

import radgpt


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(
        radgpt.utils.get_experiment_options(), case_sensitive=False
    ),
    required=True,
    help="The dataset of patient one-liners to use."
)
@click.option(
    "-l",
    "--llm",
    type=click.Choice(
        radgpt.llm.get_llm_options(), case_sensitive=False
    ),
    required=True,
    help="The base LLM model to use."
)
@click.option(
    "-m",
    "--method",
    type=str,
    required=True,
    help="The LLM query method to use."
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed."
)
@click.option(
    "--by-panel/--by-topic",
    default=True,
    show_default=True,
    help="Whether to evaluate the LLM by ACR AC panel or topic."
)
@click.option(
    "--savedir",
    type=str,
    default=None,
    show_default=True,
    help="Optional path to save the inference results to."
)
def main(
    dataset: str,
    llm: str,
    method: str,
    seed: int = 42,
    by_panel: bool = True,
    savedir: Optional[Union[Path, str]] = None,
    **kwargs
):
    """Aligning LLMs with ACR Appropriateness Criteria."""
    ac = radgpt.AppropriatenessCriteria()
    savepath = None
    _key = "panel" if by_panel else "topic"
    run_id = f"{dataset}_{llm}_{method}_{_key}_{seed}"
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        savepath = os.path.join(
            savedir, f"{run_id}.json"
        )

    # Load the specified dataset of patient one-liners.
    y_gt = radgpt.data.load_case_labels(dataset=dataset)
    patient_cases = filter(
        lambda case: radgpt.data.hashme(case) in y_gt["case"].values.tolist(),
        getattr(radgpt.data, f"read_{dataset}_dataset")()
    )
    patient_cases = list(patient_cases)
    total_case_count = len(patient_cases)

    # Instantiate the LLM client.
    llm = getattr(radgpt.llm, llm)(seed=seed)
    llm.set_system_prompt(
        radgpt.llm.DEFAULT_SYSTEM_PROMPT.format(
            ac.panels if by_panel else ac.topics,
            "Thoracic" if by_panel else "Lung Cancer Screening"
        )
    )

    batch_fn = os.environ.get("BATCH_QUERY_PATH", "batch_queries.json")
    submitted_jobs = {}
    downloaded_results = None
    if os.path.isfile(batch_fn):
        with open(batch_fn, "r") as f:
            submitted_jobs = json.load(f)
        if run_id in submitted_jobs.keys():
            downloaded_results = {}
            _batch = llm.client.batches.retrieve(submitted_jobs[run_id])
            if _batch.status == "completed":
                content = llm.client.files.content(_batch.output_file_id)
                for obj in content.text.split("\n"):
                    try:
                        run = json.loads(obj)
                    except json.JSONDecodeError:
                        continue
                    message = run["response"]["body"]["choices"][0]["message"]
                    answer = json.loads(message["content"])["answer"]
                    job_id = int(run["custom_id"].replace(run_id + "_", ""))
                    downloaded_results[job_id] = (
                        [answer] if isinstance(answer, str) else answer
                    )
            else:
                click.echo(f"Status: {_batch.status}")
                exit()

    # Evaluate the LLM according to the ACR Appropriateness Criteria.
    count, start_idx = 0, -1
    all_results = {}
    if savepath is not None and os.path.isfile(savepath):
        with open(savepath, "r") as f:
            all_results = json.load(f)
    if len(all_results.keys()):
        start_idx = max([int(k) for k in all_results.keys()])
    for idx, case in enumerate(patient_cases):
        if idx <= start_idx:
            continue
        gt = y_gt[y_gt["case"] == radgpt.data.hashme(str(case))][
            "panel" if by_panel else "topic"
        ]
        gt = gt.item().replace(", ", ",").split(",")
        if downloaded_results is not None:
            ypreds = downloaded_results[idx]
        else:
            ypreds = radgpt.llm.get_top_k_panels(
                case=str(case),
                criteria=ac,
                llm=llm,
                top_k=1,
                method=method,
                uid=f"{run_id}_{idx}"
            )
        if isinstance(ypreds, dict):
            if isinstance(all_results, dict):
                all_results = []
            all_results.append(ypreds)
        else:
            count += any([
                lbl.lower() in pred.lower() for pred in ypreds for lbl in gt
            ])
            all_results[idx] = {"ypred": ypreds, "ygt": gt}

            if savepath is not None:
                with open(savepath, "w") as f:
                    json.dump(all_results, f, indent=2)

    if isinstance(all_results, list):
        submission = llm.submit_batch_query(all_results)
        submitted_jobs[run_id] = submission.id
        with open(batch_fn, "w") as f:
            json.dump(submitted_jobs, f, indent=2)
        click.echo(submission)
    else:
        acc = count / (total_case_count - start_idx - 1)
        click.echo(f"Accuracy: {acc:.6f}")


if __name__ == "__main__":
    main()
