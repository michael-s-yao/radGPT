"""
Utility functions for running LLM model inference and additional experiments.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from transformers import AutoModel
from typing import ContextManager, Dict, Final, Sequence, Union

from .data import __all__ as data_fns


__all__ = [
    "import_flash_attn",
    "get_experiment_options",
    "split_into_sentences"
]


def import_flash_attn() -> Dict[str, Union[str, ContextManager]]:
    """
    Attempts to import and use FlashAttention for LLM model inference.
    Input:
        None.
    Returns:
        A dictionary containing the following key-value pairs:
            attn_implementation: the attention implementation to use.
            autocast_context: the corresponding autocast context manager.
    Citation(s):
        [1] Dao T, Fu DY, Ermon S, Rudra A, Re C. FlashAttention: Fast and
            memory-efficient exact attention with IO-awarness. arxiv Preprint.
            (2023). doi: 10.48550/arXiv.2205.14135
    """
    try:
        import flash_attn  # noqa
        assert torch.cuda.is_available()
        return {
            "attn_implementation": "flash_attention_2",
            "autocast_context": torch.autocast("cuda", torch.bfloat16)
        }
    except ImportError:
        return {
            "attn_implementation": "eager", "autocast_context": nullcontext()
        }


def get_experiment_options() -> Sequence[str]:
    """
    Returns a list of the implemented experiment options to run.
    Input:
        None.
    Returns:
        A list of the implemented experiment options to run.
    """
    pref, suff = "read_", "_dataset"
    options = filter(
        lambda fn: fn.startswith(pref) and fn.endswith(suff), data_fns
    )
    options = map(
        lambda fn: (
            fn.replace(pref, "", 1)[::-1].replace(suff[::-1], "", 1)[::-1]
        ),
        list(options)
    )
    return list(options)


def score(preds: Sequence[str], gts: Sequence[str]) -> bool:
    """
    Scores a set of predictions compared to ground-truth labels.
    Input:
        preds: a list of the predicted label(s).
        gts: a list of the true ground-truth label(s).
    Returns:
        Whether at least one ground-truth label is in the list of predictions.
    """
    try:
        return any([
            "".join(filter(str.isalpha, y.lower()))[:-1] in "".join(
                filter(str.isalpha, ypred.lower())
            )
            for ypred in preds for y in gts
        ])
    except Exception:
        return False


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


class NVEmbedv2(nn.Module):
    hf_repo_name: Final[str] = "nvidia/NV-Embed-v2"

    trust_remote_code: bool = True

    max_length: Final[int] = 4096

    def __init__(self, **kwargs):
        """
        Args:
            None.
        """
        super(NVEmbedv2, self).__init__()
        self.dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16

        self.model = AutoModel.from_pretrained(
            self.hf_repo_name,
            trust_remote_code=self.trust_remote_code,
            device_map="auto",
            torch_dtype=self.dtype,
            **kwargs
        )

    @torch.no_grad()
    def forward(self, inp: Union[str, Sequence[str]]) -> torch.Tensor:
        """
        Forward pass through the embedding model.
        Input:
            inp: A string or list of strings to embed.
        Returns:
            The embeddings of shape BD, where D is the embedding dimension
            and B is the number of inputs.
        """
        if isinstance(inp, str):
            inp = [inp]
        z = self.model.encode(
            inp, instruction="", max_length=self.max_length
        )
        return F.normalize(z, p=2, dim=1)

    def similarity_score(self, inp: str, corpus: torch.Tensor) -> torch.Tensor:
        """
        Computes the similarity score between an input query and a corpus of
        embeddings.
        Input:
            inp: A string query.
            corpus: The corpus of embeddings to compare the string to of shape
                BD, where D is the embedding dimension and B is the size of the
                corpus.
        Returns:
            The similarity scores as a vector of length B.
        """
        return self(inp).to(corpus) @ corpus.T
