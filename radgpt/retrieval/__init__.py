"""
Interface for retrieving relevant documents from a corpus.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import sys
from inspect import isclass
from typing import Optional, Sequence

from .base import Document, Corpus, Retriever, RandomRetriever
from .bm25 import BM25Retriever
from .bert import BERTRetriever, RadBERTRetriever
from .mpnet import MPNetRetriever
from .medcpt import MedCPTRetriever
from .cohere import CohereRetriever
from .openai import OpenAIRetriever


__all__ = [
    "Corpus",
    "Document",
    "Retriever",
    "BM25Retriever",
    "CohereRetriever",
    "OpenAIRetriever",
    "MedCPTRetriever",
    "MPNetRetriever",
    "BERTRetriever",
    "RadBERTRetriever",
    "RandomRetriever",
    "list_retrievers",
    "list_corpuses",
    "get_retriever"
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


def get_retriever(
    retriever: str,
    corpus_name: Optional[str] = None,
    corpus_dataset: Optional[np.ndarray] = None,
    **kwargs
) -> Retriever:
    """
    Retrieves the requested retriever object.
    Input:
        retriever: the name of the retriever to use.
        corpus_name: the name of the corpus dataset to load if initializing
            from a named dataset.
        corpus_dataset: the corpus dataset if initializing from an array.
    Returns:
        The requested retriever object.
    """
    corpus = Corpus(name=corpus_name, dataset=corpus_dataset)
    return getattr(sys.modules[__name__], retriever)(corpus=corpus, **kwargs)
