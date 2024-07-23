#!/usr/bin/env python3
"""
Aligning large language models (LLMs) with ACR Appropriateness Criteria.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import enlighten
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import radgpt


DATASETS_TO_PRETTY_NAMES: Dict[str, str] = {
    "synthetic": "Synthetic",
    "medbullets": "Medbullets USMLE",
    "jama_cc": "JAMA Clinical Challenges",
    "nejm": "NEJM Case Records",
    "mimic_iv": "BIDMC"
}


METHODS_TO_PRETTY_NAMES: Dict[str, str] = {
    "prompting": "Baseline",
    "icl": (
        "In-Context Learning ({icl_num_examples} Samples from {icl_retriever})"
    ),
    "rag": (
        "Retrieval-Augmented Generation "
        "({rag_top_k} Documents from {rag_corpus} using {rag_retriever})"
    ),
    "cot": "Chain-of-Thought Prompting ({cot_reasoning_method} Reasoning)"
}


def compute_imaging_results_from_topic_eval(
    topic_results: Dict[str, Dict[str, Sequence[str]]],
    ac: radgpt.AppropriatenessCriteria
) -> Dict[str, float]:
    """
    Computes the imaging statistics from an ACR AC topic evaluation.
    Input:
        topic_results: a dictionary of the ACR AC topic evaluation results.
        ac: an instance of the ACR Appropriateness Criteria.
    Returns:
        A dictionary containing the following values:
            acc: the accuracy of the imaging recommendations.
            fpr: the false positive rate corresponding to the rate at which
                unnecessary imaging studies are ordered.
            fnr: the false negative rate corresponding to the rate at which
                imaging studies were required but not obtained.
    """
    num_correct, num_cases = 0, len(topic_results.keys())
    nfp, nfn = 0, 0
    no_img_sts = ac.NO_IMAGING_INDICATION
    for _, cache_labels in topic_results.items():
        ypreds, gt = cache_labels["ypred"], cache_labels["ygt"]
        ypreds = list(
            filter(
                lambda tt: any([bool(tt in yy) for yy in ypreds]),
                ac.topics
            )
        )
        if len(ypreds) == 0:
            num_cases -= 1
            continue
        ypreds = sum(
            [ac.map_topic_to_imaging_study(yy) for yy in ypreds], []
        )
        gt = sum([ac.map_topic_to_imaging_study(yy) for yy in gt], [])
        nfn += int((ypreds == [no_img_sts]) and (no_img_sts not in gt))
        nfp += int((gt == [no_img_sts]) and (no_img_sts not in ypreds))
        num_correct += radgpt.utils.score(ypreds, gt)
    acc = 100.0 * num_correct / num_cases
    fpr, fnr = 100.0 * nfp / num_cases, 100.0 * nfn / num_cases
    return {"acc": acc, "fpr": fpr, "fnr": fnr}


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
    "--rag.top-k",
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
    "--icl.num-examples",
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
@click.option(
    "--verbose/--quiet",
    default=True,
    show_default=True,
    help="Turn on verbose outputs and logging."
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
    eval_method: str = "topic",
    savedir: Optional[Union[Path, str]] = None,
    fast_dev_run: bool = False,
    verbose: bool = True,
    **kwargs
):
    """Aligning LLMs with ACR Appropriateness Criteria."""
    logging.basicConfig(
        level=(logging.INFO if verbose else logging.CRITICAL),
        format="%(levelname)-8s %(message)s"
    )
    logger = logging.getLogger(__name__)

    ac = radgpt.AppropriatenessCriteria()
    savepath = None
    run_id = f"{dataset}_{llm}_{method}_{eval_method}_{seed}"
    if method.lower() == "rag":
        run_id += (
            f"_{rag_retriever}_{rag_corpus.replace('/', '_')}_{rag_top_k}"
        )
    elif method.lower() == "cot":
        run_id += f"_{cot_reasoning_method}"
    elif method.lower() == "icl":
        run_id += f"_{icl_retriever}_{icl_num_examples}"
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        savepath = os.path.join(
            savedir, f"{run_id}.json"
        )

    if savepath is not None and os.path.isfile(savepath):
        with open(savepath, "r") as f:
            cached_results = json.load(f)
        num_correct = 0
        for _, cache_labels in cached_results.items():
            ypreds, gt = cache_labels["ypred"], cache_labels["ygt"]
            num_correct += radgpt.utils.score(ypreds, gt)
        acc = 100.0 * num_correct / len(cached_results.keys())
        click.secho(f"Accuracy: {acc:.6f}", bold=True)

        if eval_method != "topic":
            return
        img_stats = compute_imaging_results_from_topic_eval(cached_results, ac)
        click.secho(f"Imaging Accuracy: {img_stats['acc']:.6f}", bold=True)
        click.secho(f"False Positive Rate: {img_stats['fpr']:.6f}", bold=True)
        click.secho(f"False Negative Rate: {img_stats['fnr']:.6f}", bold=True)
        return

    # Load the specified dataset of patient one-liners.
    y_gt = radgpt.data.load_case_labels(dataset=dataset)
    patient_cases = filter(
        lambda case: radgpt.data.hashme(case) in y_gt["case"].values.tolist(),
        getattr(radgpt.data, f"read_{dataset}_dataset")()
    )
    patient_cases = list(set(list(patient_cases)))

    # Instantiate the LLM client.
    llm_init_kwargs = {"seed": seed}
    if method.lower() == "rag":
        llm_init_kwargs["repetition_penalty"] = 1.5
    llm, llm_name = getattr(radgpt.llm, llm)(**llm_init_kwargs), llm
    system_prompt = radgpt.llm.get_system_prompt(
        method, rationale=cot_reasoning_method, study=(eval_method == "study")
    )
    if method.lower() == "cot":
        system_prompt = system_prompt.format(
            ac.panels
            if eval_method == "panel"
            else (ac.topics if eval_method == "topic" else ac.studies)
        )
        llm.json_format = True
        llm.max_new_tokens = 512
    else:
        system_prompt = system_prompt.format(
            ac.panels
            if eval_method == "panel"
            else (ac.topics if eval_method == "topic" else ac.studies),
            "Thoracic"
            if eval_method == "panel"
            else (
                "Lung Cancer Screening"
                if eval_method == "topic"
                else "CT chest without IV contrast screening"
            )
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
                return

    # Evaluate the LLM according to the ACR Appropriateness Criteria.
    count = 0
    all_results = {}
    with enlighten.get_manager() as mgr:
        mgr.status_bar(
            status_format=u"{llm}{fill}{method}{fill}{eval_method}",
            color="bold_underline_bright_white_on_lightslategray",
            justify=enlighten.Justify.CENTER,
            llm=llm_name,
            method=METHODS_TO_PRETTY_NAMES[method].format(
                icl_num_examples=icl_num_examples,
                icl_retriever=icl_retriever,
                cot_reasoning_method=str(cot_reasoning_method).title(),
                rag_top_k=rag_top_k,
                rag_retriever=rag_retriever,
                rag_corpus=rag_corpus
            ),
            eval_method=(
                f"ACR AC {eval_method.title()} Evaluation"
            ),
            autorefresh=True,
            min_delta=0.5
        )
        mgr.status_bar(
            status_format=u"{radgpt}{fill}{link}{fill}Seed: {seed}",
            link=mgr.term.link(
                "https://gravitas.acr.org/acportal",
                "ACR Appropriateness Criteria"
            ),
            radgpt=mgr.term.link(
                "https://github.com/michael-s-yao/radGPT", "RadGPT"
            ),
            position=1,
            fill="-",
            seed=seed,
            justify=enlighten.Justify.CENTER
        )

        terminal = mgr.term
        bar_fmt = (
            u"{desc}{desc_pad}{percentage:3.0f}%|{bar}| "
            u"C:" + mgr.term.green3(u"{count_0:{len_total}d}") + u" "
            u"I:" + mgr.term.red2(u"{count_2:{len_total}d}") + u" "
            u"E:" + terminal.yellow2(u"{count_1:{len_total}d}") + u" "
            u"[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"
        )
        with mgr.counter(
            total=len(patient_cases),
            desc=DATASETS_TO_PRETTY_NAMES[dataset.lower()],
            unit="cases",
            color='green3',
            bar_format=bar_fmt
        ) as success:
            error = success.add_subcounter("yellow2")
            failure = success.add_subcounter("red2")

            for idx, case in enumerate(patient_cases):
                gt = y_gt[y_gt["case"] == radgpt.data.hashme(str(case))][
                    "panel" if eval_method == "panel" else "topic"
                ]
                gt = re.split(r",(?=\S)", gt.item())
                if downloaded_results is not None:
                    ypreds = downloaded_results[idx]
                else:
                    rag_context, icl_context = None, None
                    if method.lower() == "rag":
                        rag_context = retriever.retrieve(
                            str(case), k=rag_top_k
                        )
                    elif method.lower() == "icl":
                        icl_context = retriever.retrieve(
                            str(case), k=icl_num_examples
                        )
                        icl_labels = []
                        for ref_case in icl_context:
                            ref_gt = y_ref[
                                y_ref["case"] == radgpt.data.hashme(
                                    str(ref_case)
                                )
                            ]
                            ref_gt = ref_gt[
                                "panel" if eval_method == "panel" else "topic"
                            ]
                            ref_gt = ref_gt.item()
                            icl_labels.append(
                                json.dumps({
                                    "answer": re.split(r",(?=\S)", ref_gt)
                                })
                            )
                        icl_context = "\n\n".join([
                            f"{ref_prompt}\n{ref_label}"
                            for ref_prompt, ref_label in zip(
                                icl_context, icl_labels
                            )
                        ])
                    ypreds = radgpt.llm.get_top_k_panels(
                        case=str(case),
                        criteria=ac,
                        llm=llm,
                        top_k=1,
                        method=method,
                        uid=f"{run_id}_{idx}",
                        rag_context=rag_context,
                        icl_context=icl_context,
                        study=(eval_method == "study")
                    )
                    if method.lower() == "cot" and (
                        cot_reasoning_method.lower() == "bayesian"
                    ):
                        ypreds = [_y[:_y.find("rationale")] for _y in ypreds]
                if isinstance(ypreds, dict):
                    if isinstance(all_results, dict):
                        all_results = []
                    all_results.append(ypreds)
                elif any(["`inf`, `nan`" in lbl for lbl in ypreds]):
                    error.update()
                    continue
                else:
                    if eval_method == "study":
                        gt = sum(
                            [ac.map_topic_to_imaging_study(yy) for yy in gt],
                            []
                        )
                    case_correct = radgpt.utils.score(ypreds, gt)
                    count += case_correct
                    if case_correct:
                        success.update()
                    else:
                        failure.update()
                    all_results[idx] = {"ypred": ypreds, "ygt": gt}

                    if savepath is not None and not fast_dev_run:
                        with open(savepath, "w") as f:
                            json.dump(all_results, f, indent=2)

                if fast_dev_run:
                    return
                logger.info(f"Case {1 + idx}: {case}")
                logger.info(f"  Predicted Label(s): {', '.join(ypreds)}")
                logger.info(f"  Ground-Truth Label(s): {', '.join(gt)}")

    if isinstance(all_results, list):
        submission = llm.submit_batch_query(all_results)
        submitted_jobs[run_id] = submission.id
        with open(batch_fn, "w") as f:
            json.dump(submitted_jobs, f, indent=2)
        click.echo(submission)
    else:
        acc = 100.0 * count / len(all_results.keys())
        click.secho(f"Accuracy: {acc:.6f}", bold=True)

        if eval_method != "topic":
            return
        img_stats = compute_imaging_results_from_topic_eval(all_results, ac)
        click.secho(f"Imaging Accuracy: {img_stats['acc']:.6f}", bold=True)
        click.secho(f"False Positive Rate: {img_stats['fpr']:.6f}", bold=True)
        click.secho(f"False Negative Rate: {img_stats['fnr']:.6f}", bold=True)


if __name__ == "__main__":
    main()
