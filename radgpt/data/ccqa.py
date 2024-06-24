"""
Reads the raw Clinical Challenge QA and MIMIC-IV datasets.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Citation(s):
    [1] Chen H, Fang Z, Singla Y, Dredze M. Benchmarking large language models
        on answering and explaining challenging medical questions. arXiv
        Preprint. (2024). doi: 10.48550/arXiv.2402.18060
    [2] Johnson AEW, Bulgarelli L, Shen L, Gayles A, Shammout A, Horng S,
        Pollard TJ, Hao S, Moody B, Gow B, Lehman LH, Celi LA, Mark RG.
        MIMIC-IV, a freely accessible electronic health record dataset. Sci
        Data 10(1). (2023). doi: 10.1038/s41597-022-01899-x

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import censusname
import numpy as np
import hashlib
import os
import pandas as pd
import random
import re
from pathlib import Path
from typing import Sequence, Union


def convert_case_to_one_liner(case: str) -> str:
    """
    Converts a patient case written in paragraph form to a single line.
    Input:
        case: the patient case.
    Returns:
        The first sentence of the patient case.
    """
    case = " ".join(case.splitlines())
    if case.count(". ") <= 1:
        return case
    return case.split(". ", 1)[0] + "."


def hashme(string: str) -> str:
    """
    Returns the SHA-512 hash of an input string.
    Input:
        string: an input string to hash.
    Returns:
        The SHA-512 hash of the input string.
    """
    hash_gen = hashlib.sha512()
    hash_gen.update(string.encode())
    return str(hash_gen.hexdigest())


def read_medbullets_dataset(
    github_branch_url: str = (
        "https://raw.githubusercontent.com/HanjieChen/ChallengeClinicalQA/"
        "dc1bc9f6923ea0ecafe2a033346b8f898e8a622c"
    ),
    medbullets_fns: Sequence[str] = [
        "medbullets/medbullets_op4.csv",
        "medbullets/medbullets_op5.csv",
    ],
    as_one_liners: bool = True
) -> np.ndarray:
    """
    Returns the Medbullets dataset of questions.
    Input:
        github_branch_url: the URL of the GitHub branch to find the Medbullets
            dataset.
        medbullets_fns: the relative filepaths of the Medbullets dataset in the
            GitHub repository.
        as_one_liners: whether to return the questions as one-liners (i.e.,
            just return the first sentence of each patient case). Default True.
    Returns:
        An array containing all the questions asked in the Medbullets dataset.
    """
    ds = np.hstack(
        list(
            map(
                lambda fn: pd.read_csv(fn)["question"].to_numpy(),
                [os.path.join(github_branch_url, fn) for fn in medbullets_fns]
            )
        )
    )
    if not as_one_liners:
        return ds
    return np.vectorize(convert_case_to_one_liner)(ds)


def read_jama_cc_dataset(
    jama_fn: Union[Path, str] = "jama_raw.csv", as_one_liners: bool = True
) -> np.ndarray:
    """
    Returns the JAMA Clinical Challenges dataset of questions. Please follow
    the instructions from @HanjieChen here to first download the dataset:
    https://github.com/HanjieChen/ChallengeClinicalQA/blob/main/README.md
    Input:
        jama_fn: the local filepath of the JAMA Clinical Challenges dataset.
        as_one_liners: whether to return the questions as one-liners (i.e.,
            just return the first sentence of each patient case). Default True.
    Returns:
        An array containing all the questions asked in the Medbullets dataset.
    """
    ds = pd.read_csv(jama_fn)["Case"].to_numpy()
    if not as_one_liners:
        return ds
    return np.vectorize(convert_case_to_one_liner)(ds)


def read_mimic_iv_dataset(
    notes_fn: Union[Path, str] = os.path.join(
        os.path.dirname(__file__), "discharge.csv.gz"
    ),
    patients_fn: Union[Path, str] = (
        "https://physionet.org/files/mimic-iv-demo/2.2/hosp/patients.csv.gz"
    ),
    hpi_title: str = "History of Present Illness:",
    seed: int = 42
) -> np.ndarray:
    """
    Returns the MIMIC-IV clinical notes dataset. Please follow the instructions
    here to first download the dataset: https://physionet.org/content/
    mimic-iv-note/2.2/
    Input:
        notes_fn: the local filepath of the MIMIC-IV `discharge.csv.gz`
            discharge notes dataset.
        patients_fn: the local filepath or URL of the MIMIC-IV
            `patients.csv.gz` file. We only use the file from the MIMIC-IV
            Demo dataset to construct our actual dataset.
        hpi_title: delimiter to indicate the start of the HPI of a clinical
            note.
        seed: random seed. Default 42.
    Returns:
        An array containing all the questions asked in the Medbullets dataset.
    """
    patients = pd.read_csv(patients_fn)
    notes = pd.read_csv(notes_fn)
    notes = notes[notes["subject_id"].isin(patients["subject_id"])]
    name = censusname.Censusname(nameformat="{surname}")
    random.seed(seed)

    one_liners = []
    for i, note in notes.iterrows():
        txt = note["text"]
        txt = txt[txt.find(hpi_title):].replace(hpi_title, "", 1)

        period = re.search(r"(?<!Mr|Dr|Ms|vs| C)\.", txt)
        if period is None:
            period = re.search(r"(?<!Mrs|Drs)\.", txt)
        if period is None:
            continue
        txt = txt[:period.start()]

        txt = txt.strip().replace("\n", " ").replace("\r", " ")
        txt = txt.replace("  ", " ")
        txt = txt + "." if txt[-1] != "." else txt

        metadata = patients[patients["subject_id"] == note["subject_id"]]
        age = metadata["anchor_age"].item()

        fill_ins = {
            r"___ (?=yo|years old|year old|year-old|y\/o|M |F )": age,
            r"(?<=Mr|Dr|Ms)\. ___": name.generate(),
            r"(?<=Mrs|Drs)\. ___": name.generate()
        }
        for regex, val in fill_ins.items():
            loc = re.search(regex, txt)
            if loc is None:
                continue
            if regex.startswith("___"):
                sidx, eidx = loc.start(), loc.start() + len("___")
            else:
                sidx, eidx = loc.end() - len("___"), loc.end()
            txt = txt[:sidx] + str(val) + txt[eidx:]

        one_liners.append(txt)
    return np.array(one_liners)
