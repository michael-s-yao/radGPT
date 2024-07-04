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
import re
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
    "--cot.reasoning_method",
    "cot_reasoning_method",
    type=click.Choice(
        radgpt.llm.get_cot_reasoning_method_options(), case_sensitive=False
    ),
    default=None,
    help="Reasoning method for chain-of-thought prompting."
)
@click.option(
    "--icl.num_examples",
    "icl_num_examples",
    type=int,
    default=1,
    show_default=True,
    help="Number of examples to use for in-context learning."
)
@click.option(
    "--icl.retriever",
    "icl_retriever",
    type=click.Choice(
        radgpt.retrieval.list_retrievers(), case_sensitive=False
    ),
    default=None,
    help="The retriever to use for finding in-context learning examples."
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
    rag_top_k: Optional[int] = 16,
    cot_reasoning_method: Optional[str] = None,
    icl_num_examples: Optional[int] = 1,
    icl_retriever: Optional[str] = None,
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
    if method.lower() == "rag":
        run_id += f"_{rag_retriever}_{rag_corpus}_{rag_top_k}"
    elif method.lower() == "cot":
        run_id += f"_{cot_reasoning_method}"
    elif method.lower() == "icl":
        run_id += f"_{icl_retriever}_{icl_num_examples}"
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
    patient_cases = list(set(list(patient_cases)))
    total_case_count = len(patient_cases)

    # Instantiate the LLM client.
    llm_init_kwargs = {"seed": seed}
    if method.lower() == "rag":
        llm_init_kwargs["repetition_penalty"] = 1.5
    llm = getattr(radgpt.llm, llm)(**llm_init_kwargs)
    system_prompt = radgpt.llm.get_system_prompt(
        method, rationale=cot_reasoning_method
    )
    if method.lower() == "cot":
        system_prompt = system_prompt.format(
            ac.panels if by_panel else ac.topics
        )
    else:
        system_prompt = system_prompt.format(
            ac.panels if by_panel else ac.topics,
            "Thoracic" if by_panel else "Lung Cancer Screening"
        )
    llm.set_system_prompt(system_prompt)

    # Load the retriever.
    if method.lower() == "rag":
        retriever = radgpt.retrieval.get_retriever(rag_retriever, rag_corpus)
    elif method.lower() == "icl":
        y_ref = radgpt.data.load_case_labels(dataset="synthetic")
        ref_ds = filter(
            lambda case: bool(
                radgpt.data.hashme(str(case)) in y_ref["case"].tolist()
            ),
            radgpt.data.read_synthetic_dataset()
        )
        retriever = radgpt.retrieval.get_retriever(
            icl_retriever, corpus_dataset=list(ref_ds)
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
    count = 0
    all_results = {}
    for idx, case in enumerate(patient_cases):
        gt = y_gt[y_gt["case"] == radgpt.data.hashme(str(case))][
            "panel" if by_panel else "topic"
        ]
        gt = re.split(r",(?=\S)", gt.item())
        if downloaded_results is not None:
            ypreds = downloaded_results[idx]
        else:
            rag_context, icl_context = None, None
            if method.lower() == "rag":
                rag_context = retriever.retrieve(str(case), k=rag_top_k)
            elif method.lower() == "icl":
                icl_context = retriever.retrieve(str(case), k=icl_num_examples)
                icl_labels = []
                for ref_case in icl_context:
                    ref_gt = y_ref[
                        y_ref["case"] == radgpt.data.hashme(str(ref_case))
                    ]
                    ref_gt = ref_gt["panel" if by_panel else "topic"].item()
                    icl_labels.append(
                        json.dumps({"answer": re.split(r",(?=\S)", ref_gt)})
                    )
                icl_context = "\n\n".join([
                    f"{ref_prompt}\n{ref_label}"
                    for ref_prompt, ref_label in zip(icl_context, icl_labels)
                ])
            ypreds = radgpt.llm.get_top_k_panels(
                case=str(case),
                criteria=ac,
                llm=llm,
                top_k=1,
                method=method,
                uid=f"{run_id}_{idx}",
                rag_context=rag_context,
                icl_context=icl_context
            )
        if isinstance(ypreds, dict):
            if isinstance(all_results, dict):
                all_results = []
            all_results.append(ypreds)
        else:
            count += any([
                lbl.lower().replace(" ", "") in pred.lower().replace(" ", "")
                for pred in ypreds for lbl in gt
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
