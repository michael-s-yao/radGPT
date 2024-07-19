import jsonlines
import os
from flask import Flask, render_template, request, send_file
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence, Union


def read_imaging_studies(
    fn: Union[Path, str] = os.path.join(
        os.path.dirname(__file__), "static", "assets", "studies.txt"
    )
) -> Sequence[str]:
    assert os.path.isfile(fn)
    with open(fn, "r") as f:
        return [study.strip() for study in f.readlines()] + ["None"]


def read_patient_cases(
    fn: Union[Path, str] = os.path.join(
        os.path.dirname(__file__), "static", "assets", "cases.txt"
    )
) -> Sequence[str]:
    assert os.path.isfile(fn)
    with open(fn, "f") as f:
        return [case.strip() for case in f.readlines()]


def write_answers():
    tmp = NamedTemporaryFile()
    with jsonlines.Writer(tmp) as writer:
        writer.write_all(request.get_json())
    tmp.seek(0)
    return send_file(tmp.name, download_name="radGPT-answers.jsonl")


def create_app() -> Flask:
    """
    Creates the Python Flask app with the relevant endpoints.
    Input:
        None.
    Returns:
        The Python Flask app instantiation with the relevant endpoints.
    """
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder=os.path.join(os.path.dirname(__file__), "templates")
    )
    os.makedirs(app.instance_path, exist_ok=True)

    app.route("/", methods=["GET"])(
        lambda: render_template(
            "index.html",
            options=read_imaging_studies(),
            questions=read_patient_cases()
        )
    )
    app.route("/submit", methods=["POST"])(write_answers)

    return app
