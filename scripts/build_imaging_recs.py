#!/usr/bin/env python3
"""
Scrapes the ACR Appropriateness Criteria to map ACR AC topics to scenarios
and corresponding imaging study appropriateness.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import jsonlines
import os
import sys
from bs4 import BeautifulSoup
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, Sequence, Union
from urllib import request
from urllib.parse import urljoin

sys.path.append(".")  # noqa
import radgpt  # noqa


def parse_table_rows_to_json(
    rows: Sequence[Sequence[str]], col_names: Sequence[str],
) -> Sequence[Dict[str, Any]]:
    """
    Parses a list of imported table rows into JSON format.
    Input:
        rows: a list of imported table rows to parse.
        col_names: a list of the column names of the table.
    Returns:
        A list of the table rows stored in JSON format.
    """
    radiation = b"\xE2\x98\xA2".decode("utf-8")
    scenario, scenario_id = None, None
    table = []
    for ridx, r in enumerate(rows):
        # Extract the scenario and scenario IDs.
        if len(r) > 4:
            scenario, scenario_id, r = r[0], r[1], r[2:]
        if scenario is None or scenario_id is None:
            continue
        # Convert string RRL values into numerical values:
        r[1] = r[1].count(radiation)
        # Insert the Peds RRL filler value if not present.
        if len(r) < 4:
            r.insert(2, None)
        else:
            r[2] = r[2].count(radiation)
        # Append the scenario and scenario IDs.
        r = [scenario, scenario_id] + r
        # Append the row to the table.
        table.append({key: val for key, val in zip(col_names, r)})

    studies = []
    for scenario, scenario_id in set(
        map(lambda r: (r[col_names[0]], r[col_names[1]]), table)
    ):
        scenario_opts = filter(
            lambda _r: _r[col_names[1]] == scenario_id, table
        )
        scenario_obj = {col_names[0]: scenario, col_names[1]: scenario_id}
        scenario_obj["Studies"] = []
        for r in scenario_opts:
            r = deepcopy(r)
            r.pop(col_names[0])
            r.pop(col_names[1])
            scenario_obj["Studies"].append(r)
        studies.append(scenario_obj)
    return studies


def main(
    savepath: Union[Path, str] = os.path.join(
        os.path.dirname(radgpt.__file__), "guidelines.jsonl"
    )
):
    """
    Scrapes the ACR Appropriateness Criteria to map ACR AC topics to scenarios
    and corresponding imaging study appropriateness.
    Input:
        savepath: path to save the data to.
    Returns:
        None.
    """
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    ac_fn = os.path.join(os.path.dirname(radgpt.__file__), "ac.json")
    with open(ac_fn, "r") as f:
        ac = json.load(f)
    base_url = ac["metadata"]["base_url"]
    guidelines = []
    for topic, _id in tqdm(ac["topics"].items()):
        html = request.urlopen(
            urljoin(base_url, f"ACPortal/GetDataForOneTopic?topicId={_id}")
        )
        topic_guidelines = []
        htmlParse = BeautifulSoup(html, "html.parser")
        for table in htmlParse.find_all("table", {"class": "tblResDocs"}):
            col_names = [
                col.get_text() for col in table.find("thead").find_all("th")
            ]
            rows = map(
                lambda r: filter(
                    lambda txt: len(txt), [txt.get_text().strip() for txt in r]
                ),
                table.find("tbody").find_all("tr")
            )
            rows = list(filter(lambda r: len(r) > 1, [list(r) for r in rows]))
            topic_guidelines += parse_table_rows_to_json(rows, col_names)
        guidelines.append({
            "Topic": topic, "Topic ID": _id, "Scenarios": topic_guidelines
        })
    with open(savepath, "w") as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(guidelines)
    return


if __name__ == "__main__":
    main()
