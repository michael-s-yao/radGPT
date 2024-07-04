"""
Utility functions for data management.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import re
from typing import Sequence


def split_into_sentences(text: str) -> Sequence[str]:
    """
    Splits an input text paragraph into its respective sentences. This
    implementation is adapted from Vladimir Fokow's original solution at
    https://stackoverflow.com/questions/4576077
    Input:
        text: the paragraph to split into sentences.
    Returns:
        A list of the sentences in the original input paragraph.
    """
    alphabets = r"([A-Za-z])"
    prefixes = r"(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = r"(Inc|Ltd|Jr|Sr|Co)"
    starters = (
        r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|"
        r"Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    )
    acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = r"[.](com|net|org|io|gov|edu|me)"
    digits = r"([0-9])"
    multiple_dots = r"\.{2,}"

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(
        multiple_dots,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text
    )
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text
    )
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    text = text.replace(".”", "”.") if "”" in text else text
    text = text.replace(".\"", "\".") if "\"" in text else text
    text = text.replace("!\"", "\"!") if "!" in text else text
    text = text.replace("?\"", "\"?") if "?" in text else text
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = [s.strip() for s in text.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences
