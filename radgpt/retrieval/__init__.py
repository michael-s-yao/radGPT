"""
Interface for retrieving relevant documents from a corpus.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import sys
from inspect import isclass
from typing import Sequence

from .base import Document, Corpus, Retriever
from .bm25 import BM25Retriever
from .bert import BERTRetriever, RadBERTRetriever


__all__ = [
    "Corpus",
    "Document",
    "Retriever",
    "BM25Retriever",
    "BERTRetriever",
    "RadBERTRetriever"
]


def list_retrievers() -> Sequence[str]:
    """
    Lists all the implemented retrievers.
    Input:
        None.
    Returns:
        A list of all the implemented retrievers.
    """
    module = sys.modules[__name__]
    opts = filter(lambda attr: isclass(getattr(module, attr)), dir(module))
    opts = filter(
        lambda attr: issubclass(getattr(module, attr), Retriever), opts
    )
    return list(filter(lambda attr: getattr(module, attr) != Retriever, opts))


def list_corpuses() -> Sequence[str]:
    """
    Lists all the available reference corpuses.
    Input:
        None.
    Returns:
        A list of the available reference corpuses.
    """
    return [
        "MedRAG/wikipedia",
        "MedRAG/statpearls",
        "MedRAG/textbooks",
        "MedRAG/pubmed",
        "michaelsyao/acrac"
    ]


def get_retriever(retriever: str, corpus_name: str, **kwargs) -> Retriever:
    """
    Retrieves the requested retriever object.
    Input:
        retriever: the name of the retriever to use.
        corpus_name: the name of the reference corpus to use.
    Returns:
        The requested retriever object.
    """
    corpus = Corpus(corpus_name)
    return getattr(sys.modules[__name__], retriever)(corpus=corpus, **kwargs)
