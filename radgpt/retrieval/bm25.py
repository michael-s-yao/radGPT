"""
BM25 retriever class implementation.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import bm25s
from typing import Sequence, Tuple, Union

from .base import Corpus, Document, Retriever


class BM25Retriever(Retriever):
    def __init__(
        self,
        corpus: Sequence[Corpus],
        method: str = "lucene",
        **kwargs
    ):
        """
        Args:
            corpus: a corpus of documents to retrieve from.
            method: the BM25 variant to use. Default `lucene`.
        """
        super(BM25Retriever, self).__init__(corpus, **kwargs)
        self.retriever = bm25s.BM25(corpus=corpus, method=method)
        self.retriever.index(bm25s.tokenize(corpus))

    def retrieve(
        self, query: str, k: int = 1, return_scores: bool = False
    ) -> Union[Sequence[Document], Tuple[Sequence[Document], Sequence[float]]]:
        """
        Retrieves the top k documents relevant to the query.
        Input:
            query: an input query.
            k: the number of documents to retrieve. Default 1.
            return_scores: whether to return the document scores.
        Retrieve:
            The top k documents relevant to the query.
        """
        results, scores = self.retriever.retrieve(bm25s.tokenize(query), k=k)
        if return_scores:
            return results[0], scores[0]
        return results[0]
