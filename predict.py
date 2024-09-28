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
    "-i",
    "--input-fn",
    required=True,
    help="A textfile of the oneliners to predict the labels for."
)
@click.option(
    "-o",
    "--output-fn",
    type=str,
    default=None,
    show_default=True,
    help="Optional path to save the inference results to."
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
    "-k",
    "--top-k",
    "--top_k",
    type=int,
    default=1,
    show_default=True,
    help="Number of predictions for the LLM to make."
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
    "--rag.top-k",
    "--rag.top_k",
    "rag_top_k",
    type=int,
    default=8,
    show_default=True,
    help="Number of documents to retrieve for RAG."
)
@click.option(
    "--cot.reasoning-method",
    "--cot.reasoning_method",
    "cot_reasoning_method",
    type=click.Choice(
        radgpt.llm.get_cot_reasoning_method_options(), case_sensitive=False
    ),
    default=None,
    help="Reasoning method for chain-of-thought prompting."
)
@click.option(
    "--icl.num-examples",
    "--icl.num_examples",
    "icl_num_examples",
    type=int,
    default=4,
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
    "--ft.model",
    "ft_model",
    type=str,
    default=None,
    help="The fine-tuned model ID or path to load for inference."
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Random seed."
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
def main(
    llm: str,
    method: str,
    top_k: int,
    input_fn: Union[Path, str],
    output_fn: Optional[Union[Path, str]] = None,
    rag_retriever: Optional[str] = None,
    rag_corpus: Optional[str] = None,
    rag_top_k: Optional[int] = 8,
    cot_reasoning_method: Optional[str] = None,
    icl_num_examples: Optional[int] = 4,
    icl_retriever: Optional[str] = None,
    ft_model: Optional[Union[Path, str]] = None,
    seed: int = 0,
    eval_method: str = "topic",
    **kwargs
):
    """Ordering diagnostic imaging using LLM-based CDS."""
    ac = radgpt.AppropriatenessCriteria()

    if os.path.isfile(input_fn):
        with open(input_fn, "r") as f:
            patient_cases = [case.strip() for case in f.readlines()]
    else:
        patient_cases = input_fn

    # Instantiate the LLM client.
    llm_init_kwargs = {"seed": seed}
    if method.lower() == "rag":
        llm_init_kwargs["repetition_penalty"] = 1.5
    if method.lower() == "ft":
        llm_init_kwargs["model_name"] = ft_model
    llm = getattr(radgpt.llm, llm)(**llm_init_kwargs)
    system_prompt = radgpt.llm.get_system_prompt(
        method, rationale=cot_reasoning_method, study=(eval_method == "study")
    )
    categories = "; ".join(
        ac.panels
        if eval_method == "panel"
        else (ac.topics if eval_method == "topic" else ac.studies)
    )
    if method.lower() == "cot":
        system_prompt = system_prompt.format(categories)
        llm.json_format = True
        llm.max_new_tokens = 512
    else:
        ex_answer = "Thoracic" if eval_method == "panel" else (
            "Lung Cancer Screening"
            if eval_method == "topic"
            else "CT chest without IV contrast screening"
        )
        system_prompt = system_prompt.format(categories, ex_answer)
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

    # Evaluate the LLM according to the ACR Appropriateness Criteria.
    answers = []
    for idx, case in enumerate(patient_cases):
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
                ref_gt = ref_gt["panel" if eval_method == "panel" else "topic"]
                ref_gt = ref_gt.item()
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
            top_k=top_k,
            method=method,
            uid=f"Q{idx}",
            rag_context=rag_context,
            icl_context=icl_context,
            study=(eval_method == "study")
        )
        click.echo(ypreds)
        if method.lower() == "cot" and (
            cot_reasoning_method.lower() == "bayesian"
        ):
            ypreds = [_y[:_y.find("rationale")] for _y in ypreds]
        elif any(["`inf`, `nan`" in lbl for lbl in ypreds]):
            ypreds = []
        answers.append(ypreds)

    if output_fn is not None:
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        with open(output_fn, "w") as f:
            f.write("\n".join(["\t".join(a) for a in answers]))


if __name__ == "__main__":
    main()
