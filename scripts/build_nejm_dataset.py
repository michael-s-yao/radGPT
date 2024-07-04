#!/usr/bin/env python3
"""
NEJM Case Records Dataset.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Citation(s):
    [1] Savage T, Nayak A, Gallo R, Rangan E, Chen JH. Diagnostic reasoning
        prompts reveal the potential for large language model
        interpretability in medicine. npj Digit Med 7(20). (2024).
        doi: 10.1038/s41746-024-01010-1

Notes:
    Please be sure to read the NEJM Terms of Use prior to running this code
    at https://www.nejmgroup.org/legal/terms-of-use.htm. Specifically, we
    highlight the following statement from their Terms of Use:

    > You may not scrape, copy, display, distribute, modify, publish,
    > reproduce, store, transmit, post, translate, or create derivative works
    > from, including with artificial intelligence tools, or in any way exploit
    > any part of the Content except that you may make use of the Content for
    > your own personal, noncommercial use, provided you keep intact all
    > copyright and other proprietary rights notices.

    Before running this script, please set your NEJM_COOKIE and USER_AGENT
    environmental variables to the `Cookie` and `User-Agent` HTML header
    parameters, respectively, when accessing the NEJM homepage at
    https://www.nejm.org/. The following website contains helpful instructions
    on where to find this information:
    https://www.scraperapi.com/blog/headers-and-cookies-for-web-scraping/

    To ensure that this script is accessible to everyone, this script only
    works if you do *not* have an active NEJM subscription or institutional
    access cached on your browser.

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import jsonlines
import os
import sys
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from urllib.request import Request, urlopen
from urllib.parse import urljoin

sys.path.append(".")  # noqa
from radgpt.data.utils import split_into_sentences  # noqa


def read_nejm_dois(
    doi_fn: Union[Path, str] = os.path.join("radgpt", "data", "nejm.txt"),
    base_url: str = "https://www.nejm.org/doi/"
) -> Sequence[Tuple[str]]:
    """
    Reads a list of input DOI's corresponding to NEJM case records.
    Input:
        doi_fn: the filepath to the text file containing the DOI's to import.
        base_url: the base URL.
    Returns:
        The list of source DOI's containing the NEJM case records of interest.
    """
    assert os.path.isfile(doi_fn)
    with open(doi_fn, "r") as f:
        return list(
            map(lambda doi: (doi, urljoin(base_url, doi)), f.readlines())
        )


def read_abstract_from_nejm_doi(
    doi_url: str, headers: Dict[str, Any] = {}
) -> str:
    """
    Reads a short abstract of a NEJM case record.
    Input:
        doi_url: the URL of the NEJM case record.
        headers: an optional dictionary of HTTP headers.
    Returns:
        The short abstract of the NEJM case record.
    """
    html = BeautifulSoup(
        urlopen(Request(doi_url, headers=headers)).read(),
        "html.parser"
    )
    abstract = html.find("section", {"id": "short-abstract"})
    try:
        return abstract.find("div", {"role": "paragraph"}).text
    except AttributeError:
        return ""


@click.command()
@click.option(
    "--dois",
    "doi_fn",
    type=str,
    default=os.path.join("radgpt", "data", "nejm.txt"),
    show_default=True,
    help="The filepath to the text file containing the DOI's to import."
)
@click.option(
    "--savepath",
    type=str,
    default="nejm.jsonl",
    show_default=True,
    help="The filepath to save the loaded dataset to."
)
def main(doi_fn: str, savepath: Optional[str] = None):
    """Builds the NEJM Case Record Dataset."""
    headers = {
        "Cookie": os.environ["NEJM_COOKIE"],
        "User-Agent": os.environ["USER_AGENT"],
    }
    now = datetime.now().astimezone().replace(microsecond=0).isoformat()

    ds = []
    for doi, doi_url in read_nejm_dois(doi_fn=doi_fn):
        abstract = read_abstract_from_nejm_doi(doi_url, headers)
        ds.append({
            "doi": doi.strip(),
            "case": " ".join(split_into_sentences(abstract)[:-1]),
            "access_date": now
        })

    if savepath is not None:
        with open(savepath, "w") as f:
            with jsonlines.Writer(f) as writer:
                writer.write_all(ds)
    else:
        click.echo(ds)


if __name__ == "__main__":
    main()
