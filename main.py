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
    type=click.Choice(radgpt.llm.get_method_options(), case_sensitive=False),
    required=True,
    help="The LLM query method to use."
)
@click.option(
    "--rag.retriever",
    "rag_retriever",
    type=click.Choice(
        radgpt.retrieval.list_retrievers(), case_sensitive=False
    ),
    default=None,
    help="The retriever to use if the query method is set to RAG."
)
@click.option(
    "--rag.corpus",
    "rag_corpus",
    type=click.Choice(radgpt.retrieval.list_corpuses(), case_sensitive=False),
    default=None,
    help="The corpus to use for the retriever if the query method is RAG."
)
@click.option(
    "--rag.top_k",
    "rag_top_k",
    type=int,
    default=16,
    show_default=True,
    help="Number of documents to retrieve for RAG."
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
@click.option(
    "--fast-dev-run",
    is_flag=True,
    show_default=True,
    default=False,
    help="Fast development run to test the implementation."
)
def main(
    dataset: str,
    llm: str,
    method: str,
    rag_retriever: Optional[str] = None,
    rag_corpus: Optional[str] = None,
    rag_top_k: Optional[int] = 32,
    seed: int = 42,
    by_panel: bool = True,
    savedir: Optional[Union[Path, str]] = None,
    fast_dev_run: bool = False,
    **kwargs
):
    """Aligning LLMs with ACR Appropriateness Criteria."""
    ac = radgpt.AppropriatenessCriteria()
    savepath = None
    _key = "panel" if by_panel else "topic"
    run_id = f"{dataset}_{llm}_{method}_{_key}_{seed}"
    if method.lower == "rag":
        run_id += f"_{rag_retriever}_{rag_corpus}_{rag_top_k}"
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
    llm_init_kwargs = {"seed": seed}
    if method.lower() == "rag":
        llm_init_kwargs["repetition_penalty"] = 1.5
    llm = getattr(radgpt.llm, llm)(**llm_init_kwargs)
    llm.set_system_prompt(
        radgpt.llm.DEFAULT_SYSTEM_PROMPT.format(
            ac.panels if by_panel else ac.topics,
            "Thoracic" if by_panel else "Lung Cancer Screening"
        )
    )

    # Load the retriever.
    if method.lower() == "rag":
        retriever = radgpt.retrieval.get_retriever(rag_retriever, rag_corpus)

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
    count = 0
    all_results = {}
    for idx, case in enumerate(patient_cases):
        gt = y_gt[y_gt["case"] == radgpt.data.hashme(str(case))][
            "panel" if by_panel else "topic"
        ]
        gt = gt.item().replace(", ", ",").split(",")
        if downloaded_results is not None:
            ypreds = downloaded_results[idx]
        else:
            rag_context = None
            if method.lower() == "rag":
                rag_context = retriever.retrieve(str(case), k=rag_top_k)
            ypreds = radgpt.llm.get_top_k_panels(
                case=str(case),
                criteria=ac,
                llm=llm,
                top_k=1,
                method=method,
                uid=f"{run_id}_{idx}",
                rag_context=rag_context
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

        if fast_dev_run:
            exit()

    if isinstance(all_results, list):
        submission = llm.submit_batch_query(all_results)
        submitted_jobs[run_id] = submission.id
        with open(batch_fn, "w") as f:
            json.dump(submitted_jobs, f, indent=2)
        click.echo(submission)
    else:
        acc = count / total_case_count
        click.echo(f"Accuracy: {acc:.6f}")


if __name__ == "__main__":
    main()
