#!/usr/bin/env python3
"""
ACR Appropriateness Criteria evidence.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import jsonlines
import os
import numpy as np
import re
import sys
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Sequence, Union
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Text, Title, Header, ListItem, NarrativeText
)
from urllib import request
from urllib.parse import urlsplit, urlunsplit

sys.path.append(".")  # noqa
from radgpt.utils import split_into_sentences  # noqa


def download_pdf(URL: str, savepath: Union[Path, str]) -> None:
    """
    Downloads a ACR AC PDF from the Internet and saves it to a specified
    directory.
    Input:
        URL: the URL source of an ACR ACR PDF. Must be of the form
            `https://acsearch.acr.org/docs/{ID}/Narrative/`, where `ID`
            is the unique document ID.
        savepath: a local file path to save the PDF to.
    Returns:
        None.
    """
    response = request.urlopen(URL)
    with open(savepath, "wb") as f:
        f.write(response.read())


def load_pdfs(
    savedir: Union[Path, str],
    acr_ac_list_source_url: str = "https://acsearch.acr.org/list",
) -> None:
    """
    Downloads all ACR AC PDFs from the Internet and saves them to a
    specified directory.
    Input:
        savedir: the directory to save the PDFs to.
        acr_ac_list_source_url: the source URL to scrape the PDFs from.
    Returns:
        None.
    """
    html = request.urlopen(acr_ac_list_source_url)
    htmlParse = BeautifulSoup(html, "html.parser")
    table = next(iter(htmlParse.find_all("div", {"class": "table"})))

    urls = table.find_all("a", {"class": "download-pdf-icon2"}, href=True)
    urls = [a["href"] for a in urls]
    urls = list(filter(lambda loc: "Narrative" in loc, urls))
    source = urlsplit(acr_ac_list_source_url)
    urls = list(
        map(
            lambda loc: urlunsplit(
                (source.scheme, source.netloc, loc, "", "")
            ),
            urls
        )
    )

    titles = table.find_all("div", {"class": "col-lg-4"})
    titles = [div.text.strip() for div in titles]

    for text, loc in tqdm(
        zip(titles, urls),
        desc="Downloading Appropriateness Criteria PDFs"
    ):
        _id = next(
            iter(filter(lambda part: part.isdigit(), loc.split("/")))
        )
        text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
        savepath = os.path.join(savedir, f"{text}_{_id}.pdf")
        if os.path.isfile(savepath):
            continue
        download_pdf(loc, savepath=savepath)


def pdf_to_chunks(
    fn: Union[Path, str],
    cachedir: Optional[Union[Path, str]] = None,
    max_len: Optional[int] = -1
) -> Sequence[Dict[str, str]]:
    """
    Parses PDFs into text chunks that are compatible with RAG.
    Input:
        fn: the filepath to the input PDF to process.
        cachedir: an optional directory to cache the chunks to.
        max_len: maximum character length for each chunk.
    Returns:
        A JSON Lines list of the extracted structured RAG data.
    """
    topic, _id = os.path.splitext(os.path.basename(fn))[0].rsplit("_", 1)
    topic = topic.replace("_", " ")

    if cachedir is not None:
        os.makedirs(cachedir, exist_ok=True)
        cache_fn = os.path.join(
            cachedir, os.path.basename(fn).split(".")[0] + ".npz"
        )

    if cachedir is not None and os.path.isfile(cache_fn):
        cache_data = np.load(cache_fn)
        content = cache_data["content"]
        title, acrids = cache_data["title"], cache_data["acrids"]
    else:
        startexp = re.compile(r"Introduction/Background\s*")
        endexp = re.compile(r"References\s*")

        elements = partition_pdf(
            filename=fn,
            strategy="hi_res",
            infer_table_structure=True,
            model_name="yolox"
        )
        content = []

        sidx, eidx = 0, len(elements)
        start = filter(
            lambda el: startexp.search(el[-1].text), enumerate(elements)
        )
        start = list(start)
        if len(start):
            sidx = start[0][0]

        end = filter(
            lambda el: endexp.search(el[-1].text), enumerate(elements)
        )
        end = list(end)
        if len(end):
            eidx = end[0][0] + 1

        elements = list(enumerate(elements[sidx:eidx]))

        titles = filter(
            lambda el: isinstance(el[-1], (Title, Header)), elements
        )
        text = filter(
            lambda el: type(el[-1]) is Text or isinstance(
                el[-1], NarrativeText
            ),
            elements
        )
        text = list(text)
        listitems = filter(
            lambda el: isinstance(el[-1], ListItem), elements
        )

        # Associate titles and headers with the next text item.
        for el_idx, el in titles:
            text_idx = el_idx + 1
            while text_idx < len(elements):
                if text_idx in [idx for idx, _ in text]:
                    break
                text_idx += 1
            if text_idx >= len(elements):
                continue
            text_match = next(
                filter(lambda text: text[0] == text_idx, text)
            )
            text_match[-1].text = el.text + " " + text_match[-1].text

        # Associate list items with the previous text item.
        for el_idx, el in listitems:
            text_idx = el_idx - 1
            while text_idx > -1:
                if text_idx in [idx for idx, _ in text]:
                    break
                text_idx -= 1
            if text_idx <= -1:
                continue
            text_match = next(
                filter(lambda text: text[0] == text_idx, text)
            )
            text_match[-1].text = text_match[-1].text + " " + el.text

        content = [t[-1].text for t in text]

        is_ascii = np.where([c.isascii() for c in content])[0]
        is_text = np.where([len(c) > 3 for c in content])[0]
        idxs = np.intersect1d(is_ascii, is_text)
        content = np.array(content)[idxs]

        if cachedir is not None:
            np.savez(cache_fn, content=content, title=title, acrids=acrids)

    if max_len > 0:
        content = sum([split_into_sentences(c) for c in content], [])
        parsed, curr = [], ""
        lim = max(max_len - len(f"{topic}. "), 1)
        for sentence in content:
            if len(curr) + len(sentence) >= lim:
                parsed.append(curr)
                curr = sentence
                continue
            curr = curr + " " + sentence
        content = parsed

    title, acrids = [topic for _ in content], [_id for _ in content]
    contents = [f"{t}. {c}" for t, c in zip(title, content)]
    ids = [f"acrac_{_id}_{idx}" for idx, _id in enumerate(acrids)]

    return [
        {"id": _id, "title": t, "content": c, "contents": tc, "ACRID": lbl}
        for _id, t, c, tc, lbl in zip(ids, title, content, contents, acrids)
    ]


@click.command()
@click.option(
    "--cache-dir",
    type=str,
    default="evidence",
    show_default=True,
    help="Directory to save the downloaded PDFs to."
)
@click.option(
    "--savedir",
    type=str,
    default=".",
    show_default=True,
    help="Directory to save the loaded dataset to."
)
@click.option(
    "--max-chunk-size",
    type=int,
    default=2048,
    show_default=True,
    help="Maximum chunk size (in characters)."
)
def main(
    cache_dir: Union[Path, str] = os.path.join(
        os.path.dirname(__file__), "evidence"
    ),
    savedir: Union[Path, str] = os.path.dirname(__file__),
    max_chunk_size: Optional[int] = 2048
):
    """Construct the ACR Appropriateness Criteria corpus for RAG."""
    os.makedirs(cache_dir, exist_ok=True)
    load_pdfs(cache_dir)

    evidence = []
    for fn in tqdm(
        list(
            filter(
                lambda fn: fn.lower().endswith(".pdf"), os.listdir(cache_dir)
            )
        )
    ):
        evidence.append(
            pdf_to_chunks(
                os.path.join(cache_dir, fn), cache_dir, max_len=max_chunk_size
            )
        )
    evidence = sum(evidence, [])

    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, "acr_evidence.jsonl"), "w") as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(evidence)


if __name__ == "__main__":
    main()
