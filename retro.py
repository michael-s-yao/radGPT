#!/usr/bin/python3
"""
Analysis script to analyze the retrospective study experimental results.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import jsonlines
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import radgpt
import scipy.stats as st
from matplotlib import font_manager, rc
from pathlib import Path
from statsmodels.stats import contingency_tables
from typing import Optional, Sequence, Tuple, Union


def get_llm_imaging_recs(
    topic_preds_fn: Union[Path, str], ac: radgpt.AppropriatenessCriteria
) -> Sequence[Sequence[str]]:
    """
    Returns the imaging recommendations based on the ACR AC topic predictions
    from an LLM for the retrospective study.
    Input:
        topic_results_fn: a filepath to the LLM's ACR AC topic predictions.
        ac: an instance of the ACR Appropriateness Criteria.
    Returns:
        A list of the corresponding LLM's imaging predictions.
    """
    with open(topic_preds_fn, "r") as f:
        with jsonlines.Reader(f) as reader:
            topics = [
                x["answer"]
                if isinstance(x["answer"], list) else [x["answer"]]
                for x in list(reader)
            ]
    studies = []
    for t in topics:
        ypreds = list(
            filter(
                lambda tt: any([bool(tt in yy) for yy in t]), ac.topics
            )
        )
        studies.append(
            sum([ac.map_topic_to_imaging_study(yy) for yy in ypreds], [])
        )
    return studies


def load_imaging_orders(
    orders_fn: Union[Path, str]
) -> Sequence[Sequence[str]]:
    """
    Loads the imaging orders from an input .jsonl file.
    Input:
        orders_fn: the filepath to an input jsonlines file of imaging orders.
    Returns:
        A list of the imaging orders from the input file.
    """
    with open(orders_fn, "r") as f:
        with jsonlines.Reader(f) as reader:
            return [x["answer"] for x in list(reader)]


def compute_accuracy_per_study_ordered(
    studies: Sequence[str], ground_truth: Sequence[str]
) -> Sequence[Sequence[int]]:
    """
    Computes whether a series of proposed imaging studies are individually
    warranted for a particular patient case.
    Input:
        studies: a list of the study(s) proposed by an algorithm for a
            particular patient scenario.
        ground_truth: a list of the ground truth correct answers for the
            patient scenario.
    Returns:
        Whether the series of proposed imaging studies are individually
        warranted according to the ground truth.
    """
    return [
        any([y in [yy.lower() for yy in ground_truth] for y in st.split(";")])
        for st in set([st.lower() for st in studies])
    ]


def compute_fp(
    all_studies: Sequence[Sequence[str]],
    ygt: Sequence[Sequence[str]],
    ac: radgpt.AppropriatenessCriteria
) -> Sequence[int]:
    """
    Computes the false positives in terms of unnecessarily ordered studies.
    Input:
        all_studies: a list of the model predictions.
        ygt: a list of the corresponding ground-truths.
        ac: an instance of the ACR Appropriateness Criteria.
    Returns:
        A list of whether there was a false positive for each of the cases.
    """
    return [
        int(
            ([ac.NO_IMAGING_INDICATION] == gt) * sum([
                ac.NO_IMAGING_INDICATION not in x for x in stu
            ])
        )
        for stu, gt in zip(all_studies, ygt)
    ]


def compute_fn(
    all_studies: Sequence[Sequence[str]],
    ygt: Sequence[Sequence[str]],
    ac: radgpt.AppropriatenessCriteria
) -> Sequence[int]:
    """
    Computes the false negatives in terms of missed imaging studies.
    Input:
        all_studies: a list of the model predictions.
        ygt: a list of the corresponding ground-truths.
        ac: an instance of the ACR Appropriateness Criteria.
    Returns:
        A list of whether there was a false negative for each of the cases.
    """
    return [
        int(
            (ac.NO_IMAGING_INDICATION not in gt) and (
                stu == [ac.NO_IMAGING_INDICATION]
            )
        )
        for stu, gt in zip(all_studies, ygt)
    ]


def compute_f1(
    all_studies: Sequence[Sequence[str]],
    ygt: Sequence[Sequence[str]],
    ac: radgpt.AppropriatenessCriteria
) -> float:
    """
    Computes the F1 score in terms of the binary task of determining
    whether imaging is required or not.
    Input:
        all_studies: a list of the model predictions.
        ygt: a list of the corresponding ground-truths.
        ac: an instance of the ACR Appropriateness Criteria.
    Returns:
        The F1 score computed over the number of patient cases.
    """
    nfn = sum(compute_fn(all_studies, ygt, ac))
    nfp = sum(compute_fp(all_studies, ygt, ac))
    ntp = 0
    for stu, gt in zip(all_studies, ygt):
        ntp += int(
            (gt != [ac.NO_IMAGING_INDICATION]) and (
                stu != [ac.NO_IMAGING_INDICATION]
            )
        )
    return float(2 * ntp) / float((2 * ntp) + nfp + nfn)


def mean_confidence_interval(
    data: Sequence[Union[float, int]], confidence_level: float = 0.95
) -> Tuple[float]:
    """
    Computes the mean confidence interval of a dataset.
    Input:
        data: a list of data values.
        confidence_level: the confidence level. Default 95%.
    Returns:
        mean: the mean of the dataset
        ll: the lower bound of the 95% confidence interval of the mean.
        ul: the upper bound of the 95% confidence interval of the mean.
    """
    data = 1.0 * np.array(data)
    mu = np.mean(data)
    ll, ul = st.t.interval(
        confidence=confidence_level,
        df=(len(data) - 1),
        loc=mu,
        scale=st.sem(data)
    )
    return mu, ll, ul


def compute_dsc(a: Sequence[str], b: Sequence[str]) -> float:
    """
    Computes the Dice similarity coefficient between two sets of
    predictions.
    Input:
        a: a list of the predictions from one algorithm.
        b: a list of the predictions from another algorithm.
    Returns:
        DSC(a, b).
    """
    a, b = list(set(a)), list(set(b))
    assert len(a) > 0 or len(b) > 0
    a = set([frozenset(item.split(";")) for item in a])
    b = set([frozenset(item.split(";")) for item in b])
    return 2.0 * len(a.intersection(b)) / float(len(a) + len(b))


def compute_mcnemar_test(
    a: Sequence[int], b: Sequence[int], thresh: int = 0
) -> contingency_tables._Bunch:
    """
    Computes the McNemar nonparameteric test between two sets of predictions.
    Input:
        a: a list of the predictions from one algorithm.
        b: a list of the predictions from another algorithm.
    Returns:
        The results of the McNemar test.
    """
    table = [[0, 0], [0, 0]]
    for aa, bb in zip(a, b):
        table[1 - (aa > thresh)][1 - (bb > thresh)] += 1
    correction = np.any(np.array(table) <= 4)
    return contingency_tables.mcnemar(table, exact=True, correction=correction)


@click.command()
@click.option(
    "--results-dir",
    "-r",
    type=str,
    default="docs/retrospective",
    show_default=True,
    help="Path to the directory of retrospective results."
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=0,
    show_default=True,
    help="Random seed."
)
@click.option(
    "--savedir",
    type=str,
    default=".",
    show_default=True,
    help="Optional path to the directory to save the plots to."
)
def main(
    results_dir: Union[Path, str] = "docs/retrospective",
    seed: Optional[int] = 0,
    savedir: Union[Path, str] = "."
):
    """Analyzes the retrospective study experimental results."""
    ac = radgpt.AppropriatenessCriteria()
    gt = load_imaging_orders(
        os.path.join(results_dir, "ground_truth.jsonl")
    )
    doctor = load_imaging_orders(
        os.path.join(results_dir, "clinician.jsonl")
    )
    doctor_scores = [
        compute_accuracy_per_study_ordered(doc, ygt)
        for doc, ygt in zip(doctor, gt)
    ]
    doctor_fps = compute_fp(doctor, gt, ac)
    doctor_fns = compute_fn(doctor, gt, ac)
    doctor_color = "#272319"

    claude_accs, claude_fps, claude_fns, claude_f1s = [], [], [], []
    claude_mu, claude_ll, claude_ul = [], [], []
    claude_color = "#204D8C"
    llama_accs, llama_fps, llama_fns, llama_f1s = [], [], [], []
    llama_mu, llama_ll, llama_ul = [], [], []
    llama_color = "#3B6F38"
    all_claude_llama_dscs = []
    all_claude_doc_dscs = []
    all_llama_doc_dscs = []
    for k in range(4):
        claude = get_llm_imaging_recs(
            os.path.join(results_dir, f"ClaudeSonnet/{k + 1}.jsonl"), ac
        )
        llama = get_llm_imaging_recs(
            os.path.join(results_dir, f"Llama3Instruct/{k + 1}.jsonl"), ac
        )

        claude_scores, llama_scores = [], []
        claude_llama_dscs, claude_doc_dscs, llama_doc_dscs = [], [], []
        for cla, lla, doc, ygt in zip(claude, llama, doctor, gt):
            claude_scores.append(compute_accuracy_per_study_ordered(cla, ygt))
            llama_scores.append(compute_accuracy_per_study_ordered(lla, ygt))

            claude_llama_dscs.append(compute_dsc(cla, lla))
            claude_doc_dscs.append(compute_dsc(cla, doc))
            llama_doc_dscs.append(compute_dsc(lla, doc))
        all_claude_llama_dscs.append(claude_llama_dscs)
        all_claude_doc_dscs.append(claude_doc_dscs)
        all_llama_doc_dscs.append(llama_doc_dscs)
        print(
            f"(LLaMA vs Claude) vs (LLama vs Doctor) DSC (k@{k + 1}):",
            st.ttest_ind(
                claude_llama_dscs,
                llama_doc_dscs,
                random_state=seed,
                alternative="two-sided"
            )
        )
        print(
            f"(LLaMA vs Claude) vs (Claude vs Doctor) DSC (k@{k + 1}):",
            st.ttest_ind(
                claude_llama_dscs,
                claude_doc_dscs,
                random_state=seed,
                alternative="two-sided"
            )
        )
        print(
            f"(LLaMA vs Doctor) vs (Claude vs Doctor) DSC (k@{k + 1}):",
            st.ttest_ind(
                llama_doc_dscs,
                claude_doc_dscs,
                random_state=seed,
                alternative="two-sided"
            )
        )

        llama_accs.append(
            sum(sum(llama_scores, [])) / len(sum(llama_scores, []))
        )
        llama_fps.append(compute_fp(llama, gt, ac))
        llama_fns.append(compute_fn(llama, gt, ac))
        llama_f1s.append(compute_f1(llama, gt, ac))
        print(f"LLaMA F1 Score (k@{k + 1}):", llama_f1s[-1])
        print(
            f"LLaMA FPR (k@{k + 1}):",
            sum(llama_fps[-1]) / float(len(llama_fps[-1])),
            compute_mcnemar_test(llama_fps[-1], doctor_fps, thresh=3)
        )
        llama_fps[-1] = sum(llama_fps[-1]) / float(len(llama_fps[-1]))
        print(
            f"LLaMA FNR (k@{k + 1}):",
            sum(llama_fns[-1]) / float(len(llama_fns[-1])),
            compute_mcnemar_test(llama_fns[-1], doctor_fns)
        )
        llama_fns[-1] = sum(llama_fns[-1]) / float(len(llama_fns[-1]))
        print(
            f"LLaMA Accuracy (k@{k + 1}):",
            llama_accs[-1],
            compute_mcnemar_test(
                [int(sum(x) > 0) for x in llama_scores],
                [int(sum(x) > 0) for x in doctor_scores]
            )
        )
        mu, ll, ul = mean_confidence_interval([len(lla) for lla in llama])
        llama_mu.append(mu)
        llama_ll.append(ll)
        llama_ul.append(ul)
        print(f"LLaMA Number of Studies (k@{k + 1}): {mu} [{ll}, {ul}]")
        print(
            f"LLaMA vs Doctor Number of Studies (k@{k + 1}):",
            st.ttest_ind(
                [len(lla) for lla in llama],
                [len(doc) for doc in doctor],
                random_state=seed,
                alternative="two-sided"
            )
        )

        claude_accs.append(
            sum(sum(claude_scores, [])) / len(sum(claude_scores, []))
        )
        claude_fps.append(compute_fp(claude, gt, ac))
        claude_fns.append(compute_fn(claude, gt, ac))
        claude_f1s.append(compute_f1(claude, gt, ac))
        print(f"Claude F1 Score (k@{k + 1}):", claude_f1s[-1])
        print(
            f"Claude FPR (k@{k + 1}):",
            sum(claude_fps[-1]) / float(len(claude_fps[-1])),
            compute_mcnemar_test(claude_fps[-1], doctor_fps, thresh=3)
        )
        claude_fps[-1] = sum(claude_fps[-1]) / float(len(claude_fps[-1]))
        print(
            f"Claude FNR (k@{k + 1}):",
            sum(claude_fns[-1]) / float(len(claude_fns[-1])),
            compute_mcnemar_test(claude_fns[-1], doctor_fns)
        )
        claude_fns[-1] = sum(claude_fns[-1]) / float(len(claude_fns[-1]))
        print(
            f"Claude Accuracy (k@{k + 1}):",
            claude_accs[-1],
            compute_mcnemar_test(
                [int(sum(x) > 0) for x in claude_scores],
                [int(sum(x) > 0) for x in doctor_scores]
            )
        )
        mu, ll, ul = mean_confidence_interval([len(cla) for cla in claude])
        claude_mu.append(mu)
        claude_ll.append(ll)
        claude_ul.append(ul)
        print(f"Claude Number of Studies (k@{k + 1}): {mu} [{ll}, {ul}]")
        print(
            f"Claude vs Doctor Number of Studies (k@{k + 1}):",
            st.ttest_ind(
                [len(cla) for cla in claude],
                [len(doc) for doc in doctor],
                random_state=seed,
                alternative="two-sided"
            )
        )

        if k == 0:
            doctor_acc = sum(sum(doctor_scores, [])) / len(
                sum(doctor_scores, [])
            )
            mpl.rcParams["axes.linewidth"] = 1.5
            mpl.rcParams["font.family"] = "Arial"
            mpl.rcParams["font.size"] = 12
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
            ax = ax.flatten()
            ax[0].text(
                -0.1,
                1.1,
                "(A)",
                transform=ax[0].transAxes,
                size=14,
                weight="bold"
            )
            ax[0].bar(
                ["Llama 3", "Claude Sonnet-3.5", "Physician"],
                100.0 * np.array([llama_accs[0], claude_accs[0], doctor_acc]),
                color=[
                    llama_color + "aa",
                    claude_color + "aa",
                    doctor_color + "aa"
                ],
                hatch=["/", "\\", ""],
                edgecolor="k",
                linewidth=2
            )
            ax[0].set_xticklabels(
                ["Llama 3", "Claude\nSonnet-3.5", "Physician"], rotation=30
            )
            ax[0].xaxis.set_tick_params(width=1.5)
            ax[0].yaxis.set_tick_params(width=1.5)
            ax[0].set_ylabel("Accuracy", fontweight="bold")
            ax[0].spines[["right", "top"]].set_visible(False)

            doctor_fpr = sum(doctor_fps) / float(len(doctor_fps))
            ax[1].text(
                -0.1,
                1.1,
                "(B)",
                transform=ax[1].transAxes,
                size=14,
                weight="bold"
            )
            ax[1].bar(
                ["Llama 3", "Claude Sonnet-3.5", "Physician"],
                100.0 * np.array([
                    llama_fps[0], claude_fps[0], doctor_fpr
                ]),
                color=[
                    llama_color + "aa",
                    claude_color + "aa",
                    doctor_color + "aa"
                ],
                hatch=["/", "\\", ""],
                edgecolor="k",
                linewidth=2
            )
            ax[1].set_xticklabels(
                ["Llama 3", "Claude\nSonnet-3.5", "Physician"], rotation=30
            )
            ax[1].xaxis.set_tick_params(width=1.5)
            ax[1].yaxis.set_tick_params(width=1.5)
            ax[1].set_ylabel("False Positive Rate", fontweight="bold")
            ax[1].spines[["right", "top"]].set_visible(False)

            doctor_fnr = sum(doctor_fns) / float(len(doctor_fns))
            ax[2].text(
                -0.1,
                1.1,
                "(C)",
                transform=ax[2].transAxes,
                size=14,
                weight="bold"
            )
            ax[2].bar(
                ["Llama 3", "Claude Sonnet-3.5", "Physician"],
                100.0 * np.array([llama_fns[0], claude_fns[0], doctor_fnr]),
                color=[
                    llama_color + "aa",
                    claude_color + "aa",
                    doctor_color + "aa"
                ],
                hatch=["/", "\\", ""],
                edgecolor="k",
                linewidth=2
            )
            ax[2].set_xticklabels(
                ["Llama 3", "Claude\nSonnet-3.5", "Physician"], rotation=30
            )
            ax[2].xaxis.set_tick_params(width=1.5)
            ax[2].yaxis.set_tick_params(width=1.5)
            ax[2].set_ylabel("False Negative Rate", fontweight="bold")
            ax[2].spines[["right", "top"]].set_visible(False)

            doctor_f1 = compute_f1(doctor, gt, ac)
            ax[3].text(
                -0.1,
                1.1,
                "(D)",
                transform=ax[3].transAxes,
                size=14,
                weight="bold"
            )
            ax[3].bar(
                ["Llama 3", "Claude Sonnet-3.5", "Physician"],
                100.0 * np.array([llama_f1s[0], claude_f1s[0], doctor_f1]),
                color=[
                    llama_color + "aa",
                    claude_color + "aa",
                    doctor_color + "aa"
                ],
                hatch=["/", "\\", ""],
                edgecolor="k",
                linewidth=2
            )
            ax[3].set_xticklabels(
                ["Llama 3", "Claude\nSonnet-3.5", "Physician"], rotation=30
            )
            ax[3].xaxis.set_tick_params(width=1.5)
            ax[3].yaxis.set_tick_params(width=1.5)
            ax[3].set_ylabel(r"$\mathregular{F_1}$ Score", fontweight="bold")
            ax[3].spines[["right", "top"]].set_visible(False)

            doctor_mu, ll, ul = mean_confidence_interval([
                len(doc) for doc in doctor
            ])
            ax[4].text(
                -0.1, 1.1,
                "(E)",
                transform=ax[4].transAxes,
                size=14,
                weight="bold"
            )
            ax[4].bar(
                ["Llama 3", "Claude Sonnet-3.5", "Physician"],
                np.array([llama_mu[0], claude_mu[0], doctor_mu]),
                color=[
                    llama_color + "aa",
                    claude_color + "aa",
                    doctor_color + "aa"
                ],
                hatch=["/", "\\", ""],
                edgecolor="k",
                linewidth=2
            )
            ax[4].errorbar(
                ["Llama 3", "Claude Sonnet-3.5", "Physician"],
                np.array([llama_mu[0], claude_mu[0], doctor_mu]),
                yerr=[
                    llama_mu[0] - llama_ll[0],
                    claude_mu[0] - claude_ll[0],
                    doctor_mu - ll
                ],
                color="k",
                elinewidth=1.5,
                linewidth=0,
                capsize=2,
                capthick=1.5
            )
            ax[4].set_xticklabels(
                ["Llama 3", "Claude\nSonnet-3.5", "Physician"], rotation=30
            )
            ax[4].xaxis.set_tick_params(width=1.5)
            ax[4].yaxis.set_tick_params(width=1.5)
            ax[4].set_ylabel(
                "Number of Imaging\nStudies Recommended", fontweight="bold"
            )
            ax[4].spines[["right", "top"]].set_visible(False)

            _llama_mu, _llama_ll, _ = mean_confidence_interval(
                all_llama_doc_dscs[0]
            )
            _claude_mu, _claude_ll, _ = mean_confidence_interval(
                all_claude_doc_dscs[0]
            )
            _llm_mu, _llm_ll, _ = mean_confidence_interval(
                all_claude_llama_dscs[0]
            )
            ax[5].text(
                -0.1,
                1.1,
                "(F)",
                transform=ax[5].transAxes,
                size=14,
                weight="bold"
            )
            ax[5].bar(
                [
                    "Llama 3 and Physician Agreement",
                    "Claude Sonnet-3.5 and Physician Agreement",
                    "Llama 3 and Claude Sonnet-3.5 Agreement"
                ],
                100.0 * np.array([_llama_mu, _claude_mu, _llm_mu]),
                color=["#2C6677" + "aa", "#AA5621" + "aa", "#703171" + "aa"],
                hatch=["/", "\\", ""],
                edgecolor="k",
                linewidth=2
            )
            ax[5].errorbar(
                [
                    "Llama 3 and Physician Agreement",
                    "Claude Sonnet-3.5 and Physician Agreement",
                    "Llama 3 and Claude Sonnet-3.5 Agreement"
                ],
                100.0 * np.array([_llama_mu, _claude_mu, _llm_mu]),
                yerr=100.0 * np.array([
                    _llama_mu - _llama_ll,
                    _claude_mu - _claude_ll,
                    _llm_mu - _llm_ll
                ]),
                color="k",
                elinewidth=1.5,
                linewidth=0,
                capsize=2,
                capthick=1.5
            )
            ax[5].set_xticklabels(
                [
                    "Llama 3 and\nPhysician Agreement",
                    "Claude Sonnet-3.5 and\nPhysician Agreement",
                    "Llama 3 and Claude\nSonnet-3.5 Agreement"
                ],
                rotation=30,
            )
            ax[5].xaxis.set_tick_params(width=1.5)
            ax[5].yaxis.set_tick_params(width=1.5)
            ax[5].set_ylabel("Dice-Sørensen Coefficient", fontweight="bold")
            ax[5].spines[["right", "top"]].set_visible(False)

            fig.tight_layout()
            plt.savefig(
                os.path.join(savedir, "main.svg"),
                transparent=True,
                bbox_inches="tight",
                dpi=600
            )

    plt.figure(figsize=(5, 3))

    doctor_acc = sum(sum(doctor_scores, [])) / len(sum(doctor_scores, []))
    plt.plot(
        np.arange(len(claude_accs)) + 1,
        100.0 * np.array(claude_accs),
        "-^",
        color=claude_color,
        label="Claude Sonnet-3.5"
    )
    plt.plot(
        np.arange(len(llama_accs)) + 1,
        100.0 * np.array(llama_accs),
        ":D",
        color=llama_color,
        label="Llama 3"
    )
    plt.plot(
        np.arange(len(llama_accs)) + 1,
        100.0 * np.ones(len(llama_accs)) * doctor_acc,
        "--",
        color=doctor_color,
        alpha=0.5,
        label="Physician"
    )
    plt.xlabel("Maximum Number of Predicted ACR AC Topics", fontweight="bold")
    plt.xticks(np.arange(len(llama_accs)) + 1)
    plt.ylabel("Accuracy", fontweight="bold")
    plt.legend(loc="upper right")
    if savedir is not None:
        plt.savefig(
            os.path.join(savedir, "acc.svg"),
            dpi=600,
            bbox_inches="tight",
            transparent=True
        )
    else:
        plt.show()
    print("Doctor Acc:", doctor_acc)
    plt.cla()

    doctor_f1 = compute_f1(doctor, gt, ac)
    plt.plot(
        np.arange(len(claude_f1s)) + 1,
        100.0 * np.array(claude_f1s),
        "-^",
        color=claude_color,
        label="Claude Sonnet-3.5"
    )
    plt.plot(
        np.arange(len(llama_f1s)) + 1,
        100.0 * np.array(llama_f1s),
        ":D",
        color=llama_color,
        label="Llama 3"
    )
    plt.plot(
        np.arange(len(llama_f1s)) + 1,
        100.0 * np.ones(len(llama_f1s)) * doctor_f1,
        "--",
        color=doctor_color,
        alpha=0.5,
        label="Physician"
    )
    plt.xlabel("Maximum Number of Predicted ACR AC Topics", fontweight="bold")
    plt.xticks(np.arange(len(llama_f1s)) + 1)
    plt.ylabel(r"$\mathregular{F_1}$ Score", fontweight="bold")
    plt.legend(loc="upper right")
    if savedir is not None:
        plt.savefig(
            os.path.join(savedir, "f1.svg"),
            dpi=600,
            bbox_inches="tight",
            transparent=True
        )
    else:
        plt.show()
    print("Doctor F1 Score:", doctor_f1)
    plt.cla()

    doctor_fpr = sum(doctor_fps) / float(len(doctor_fps))
    plt.plot(
        np.arange(len(claude_fps)) + 1,
        100.0 * np.array(claude_fps),
        "-^",
        color=claude_color,
        label="Claude Sonnet-3.5"
    )
    plt.plot(
        np.arange(len(llama_fps)) + 1,
        100.0 * np.array(llama_fps),
        ":D",
        color=llama_color,
        label="Llama 3"
    )
    plt.plot(
        np.arange(len(llama_fps)) + 1,
        100.0 * np.ones(len(llama_fps)) * doctor_fpr,
        "--",
        color=doctor_color,
        alpha=0.5,
        label="Physician"
    )
    plt.xlabel("Maximum Number of Predicted ACR AC Topics", fontweight="bold")
    plt.xticks(np.arange(len(llama_fps)) + 1)
    plt.ylabel("False Positive Rate", fontweight="bold")
    plt.legend(loc="upper left")
    if savedir is not None:
        plt.savefig(
            os.path.join(savedir, "fpr.svg"),
            dpi=600,
            bbox_inches="tight",
            transparent=True
        )
    else:
        plt.show()
    print("Doctor FPR:", doctor_fpr)
    plt.cla()

    doctor_fnr = sum(doctor_fns) / float(len(doctor_fns))
    plt.plot(
        np.arange(len(claude_fns)) + 1,
        100.0 * np.array(claude_fns),
        "-^",
        color=claude_color,
        label="Claude Sonnet-3.5"
    )
    plt.plot(
        np.arange(len(llama_fns)) + 1,
        100.0 * np.array(llama_fns),
        ":D",
        color=llama_color,
        label="Llama 3"
    )
    plt.plot(
        np.arange(len(llama_fns)) + 1,
        100.0 * np.ones(len(llama_fns)) * doctor_fnr,
        "--",
        color=doctor_color,
        alpha=0.5,
        label="Physician"
    )
    plt.xlabel("Maximum Number of Predicted ACR AC Topics", fontweight="bold")
    plt.xticks(np.arange(len(llama_fns)) + 1)
    plt.ylabel("False Negative Rate", fontweight="bold")
    plt.legend(loc="upper right")
    if savedir is not None:
        plt.savefig(
            os.path.join(savedir, "fnr.svg"),
            dpi=600,
            bbox_inches="tight",
            transparent=True
        )
    else:
        plt.show()
    print("Doctor FNR:", doctor_fnr)
    plt.cla()

    plt.plot(
        np.arange(len(claude_accs)) + 1,
        100.0 * np.array(claude_accs),
        "-^",
        color=claude_color,
        label="Claude Sonnet-3.5"
    )
    plt.plot(
        np.arange(len(llama_accs)) + 1,
        100.0 * np.array(llama_accs),
        ":D",
        color=llama_color,
        label="Llama 3"
    )
    plt.plot(
        np.arange(len(llama_accs)) + 1,
        100.0 * np.ones(len(llama_accs)) * doctor_acc,
        "--",
        color=doctor_color,
        alpha=0.5,
        label="Physician"
    )
    plt.xlabel("Maximum Number of Predicted ACR AC Topics", fontweight="bold")
    plt.xticks(np.arange(len(llama_accs)) + 1)
    plt.ylabel("Accuracy", fontweight="bold")
    plt.legend(loc="upper right")
    if savedir is not None:
        plt.savefig(
            os.path.join(savedir, "acc.svg"),
            dpi=600,
            bbox_inches="tight",
            transparent=True
        )
    else:
        plt.show()
    plt.cla()

    mu, ll, ul = mean_confidence_interval([len(doc) for doc in doctor])
    plt.fill_between(
        np.arange(len(llama_mu)) + 1,
        np.ones(len(llama_mu)) * ll,
        np.ones(len(llama_mu)) * ul,
        linewidth=0,
        color=doctor_color,
        alpha=0.2
    )
    plt.fill_between(
        np.arange(len(claude_mu)) + 1,
        np.array(claude_ll),
        np.array(claude_ul),
        linewidth=0,
        color=claude_color,
        alpha=0.2
    )
    plt.fill_between(
        np.arange(len(llama_mu)) + 1,
        np.array(llama_ll),
        np.array(llama_ul),
        linewidth=0,
        color=llama_color,
        alpha=0.2
    )
    plt.plot(
        np.arange(len(llama_mu)) + 1,
        np.ones(len(llama_mu)) * mu,
        "--",
        alpha=0.5,
        color=doctor_color,
        label="Physician"
    )
    plt.plot(
        np.arange(len(claude_mu)) + 1,
        np.array(claude_mu),
        "-^",
        color=claude_color,
        label="Claude Sonnet-3.5"
    )
    plt.plot(
        np.arange(len(llama_mu)) + 1,
        np.array(llama_mu),
        ":D",
        color=llama_color,
        label="Llama 3"
    )
    plt.xlabel("Maximum Number of Predicted ACR AC Topics", fontweight="bold")
    plt.xticks(np.arange(len(llama_accs)) + 1)
    plt.ylabel("Number of Imaging Studies Recommended", fontweight="bold")
    plt.legend(loc="upper left")
    if savedir is not None:
        plt.savefig(
            os.path.join(savedir, "num.svg"),
            dpi=600,
            bbox_inches="tight",
            transparent=True
        )
    else:
        plt.show()
    print(f"Doctor Number of Studies: {mu} [{ll}, {ul}]")
    plt.cla()

    plt.fill_between(
        np.arange(len(all_claude_llama_dscs)) + 1,
        [
            100.0 * mean_confidence_interval(x)[1]
            for x in all_claude_llama_dscs
        ],
        [
            100.0 * mean_confidence_interval(x)[2]
            for x in all_claude_llama_dscs
        ],
        linewidth=0,
        color="#703171",
        alpha=0.2
    )
    plt.fill_between(
        np.arange(len(all_claude_doc_dscs)) + 1,
        [100.0 * mean_confidence_interval(x)[1] for x in all_claude_doc_dscs],
        [100.0 * mean_confidence_interval(x)[2] for x in all_claude_doc_dscs],
        linewidth=0,
        color="#AA5621",
        alpha=0.2
    )
    plt.fill_between(
        np.arange(len(all_claude_doc_dscs)) + 1,
        [100.0 * mean_confidence_interval(x)[1] for x in all_llama_doc_dscs],
        [100.0 * mean_confidence_interval(x)[2] for x in all_llama_doc_dscs],
        linewidth=0,
        color="#2C6677",
        alpha=0.2
    )
    plt.plot(
        np.arange(len(all_claude_llama_dscs)) + 1,
        [
            100.0 * mean_confidence_interval(x)[0]
            for x in all_claude_llama_dscs
        ],
        ":D",
        color="#703171",
        label="Claude Sonnet-3.5 and Llama 3 Agreement"
    )
    plt.plot(
        np.arange(len(all_claude_doc_dscs)) + 1,
        [100.0 * mean_confidence_interval(x)[0] for x in all_claude_doc_dscs],
        ":D",
        color="#AA5621",
        label="Claude Sonnet-3.5 and Physician Agreement"
    )
    plt.plot(
        np.arange(len(all_llama_doc_dscs)) + 1,
        [100.0 * mean_confidence_interval(x)[0] for x in all_llama_doc_dscs],
        ":D",
        color="#2C6677",
        label="Llama 3 and Physician Agreement"
    )
    plt.xlabel("Maximum Number of Predicted ACR AC Topics", fontweight="bold")
    plt.xticks(np.arange(len(all_claude_llama_dscs)) + 1)
    plt.ylabel("Dice-Sørensen Coefficient", fontweight="bold")
    plt.ylim(0, 80)
    plt.legend()
    if savedir is not None:
        plt.savefig(
            os.path.join(savedir, "dsc.svg"),
            dpi=600,
            bbox_inches="tight",
            transparent=True
        )
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    font_dirs = [os.path.join(os.environ["HOME"], ".cache", "msttcorefonts")]
    for ff in font_manager.findSystemFonts(fontpaths=font_dirs):
        font_manager.fontManager.addfont(ff)
    rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})
    main()
