#!/usr/bin/env python3
"""
Flask app for human-user studies.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import json
import jsonlines
import logging
import os
from collections import defaultdict
from datetime import datetime
from flask import Flask, render_template, redirect, request, url_for
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path
from typing import Any, Dict, Sequence, Union


def read_imaging_studies(
    fn: Union[Path, str] = os.path.join(
        os.path.dirname(__file__), "static", "assets", "studies.txt"
    )
) -> Sequence[str]:
    """
    Imports the space of valid imaging studies to choose from.
    Input:
        fn: the path to the file containing the listed imaging studies.
    Returns:
        A list of the valid imagng studies to choose from.
    """
    assert os.path.isfile(fn)
    with open(fn, "r") as f:
        return [study.strip() for study in f.readlines()] + ["None"]


def read_patient_cases(
    fn: Union[Path, str] = os.path.join(
        os.path.dirname(__file__), "static", "assets", "cases.jsonl"
    )
) -> Sequence[str]:
    """
    Imports the patient cases to evaluate.
    Input:
        fn: the path to the file containing the patient cases.
    Returns:
        A list of the patient cases to evaluate.
    """
    assert os.path.isfile(fn)
    with open(fn, "r") as f:
        with jsonlines.Reader(f) as reader:
            return list(reader)


def read_guidelines(
    fn: Union[Path, str] = os.path.join(
        os.path.dirname(__file__), "static", "assets", "guidelines.jsonl"
    )
) -> Sequence[Dict[str, Any]]:
    """
    Reads the ACR Appropriateness Criteria guidelines.
    Input:
        fn: the path to the file containing the imaging guidelines.
    Returns:
        A list of the guideline objects.
    """
    with open(fn, "r") as f:
        with jsonlines.Reader(f) as reader:
            return list(reader)


def create_app(debug: bool = False) -> Flask:
    """
    Creates the Python Flask app with the relevant endpoints.
    Input:
        debug: whether the function is run in the development environment.
    Returns:
        The Python Flask app instantiation with the relevant endpoints.
    """
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder=os.path.join(os.path.dirname(__file__), "templates")
    )
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
    os.makedirs(app.instance_path, exist_ok=True)

    db = SQLAlchemy(app)

    class User(db.Model):
        __tablename__ = "results"

        uid = db.Column(db.Integer, primary_key=True)
        time = db.Column(db.String(128), index=True)
        response = db.Column(db.String(2048), index=True)

    cases = read_patient_cases(
        os.path.join(
            os.path.dirname(__file__),
            "static",
            "assets",
            "demo.jsonl" if debug else "cases.jsonl"
        )
    )
    questions = list(map(lambda data: data["case"], cases))
    topics = list(map(lambda data: data["topics"], cases))

    category_to_format = defaultdict(lambda: "gray")
    category_to_format.update({
        "Usually appropriate": "green",
        "May be appropriate": "yellow",
        "Usually not appropriate": "red",
    })

    guidelines = {
        data["Topic"]: [
            {
                "Procedure": study["Procedure"],
                "Radiation": study["Adult RRL"],
                "Category": category_to_format[
                    study["Appropriateness Category"]
                ]
            }
            for study in data["Scenarios"][0]["Studies"]
        ]
        for data in read_guidelines()
    }
    guidelines = [
        [{"Topic": t, "Table": guidelines[t]} for t in topic_list]
        for topic_list in topics
    ]

    @app.route("/", methods=["GET"])
    def index():
        uid = request.args.get("uid", None)
        assert uid is not None
        return render_template(
            "index.html",
            options=read_imaging_studies(),
            questions=questions,
            guidelines=guidelines,
            uid=uid
        )

    @app.route("/success", methods=["GET"])
    def success():
        return render_template("success.html")

    @app.route("/api/v1/submit", methods=["POST"])
    def write_answers():
        uid = request.form.pop("uid", "None")
        response = [
            {"question": question, "answer": answer}
            for question, answer in request.form
            if question != "uid"
        ]
        time = datetime.now().astimezone().replace(microsecond=0).isoformat()
        user = User(uid=uid, time=time, response=json.dumps(response))
        db.session.add(user)
        try:
            db.session.commit()
        except Exception as e:
            logging.error(e)
        return redirect(url_for("success"))

    with app.app_context():
        db.create_all()

    return app


if __name__ == "__main__":
    debug = True
    create_app(debug=debug).run(debug=debug)
