"""
OpenAI embedding  model-based retriever.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import faiss
import os
import numpy as np
from openai import OpenAI
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .base import Corpus, Document, Retriever


class OpenAIRetriever(Retriever):
    model_name: str = "text-embedding-3-large"

    hidden_size: int = 3072

    def __init__(
        self,
        corpus: Corpus,
        index_dir: Optional[Union[Path, str]] = os.path.join(
            os.path.dirname(__file__), "indices"
        ),
        **kwargs
    ):
        """
        Args:
            corpus: a corpus of documents to retrieve from.
            index_dir: the cache path to load and save the index from.
        """
        super(OpenAIRetriever, self).__init__(
            corpus=corpus, index_dir=index_dir, **kwargs
        )
        self.client = OpenAI()

        if os.path.isfile(self.index_fn):
            self.index = faiss.read_index(self.index_fn)
            return
        self.index = faiss.IndexFlatIP(self.hidden_size)
        for vec in [self.embed(doc.text) for doc in self.corpus]:
            self.index.add(vec[np.newaxis])
        faiss.write_index(self.index, self.index_fn)

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
        embedding = self.embed(query)[np.newaxis]
        results = self.index.search(embedding, k=k)
        scores, idxs = self.index.search(embedding, k=k)
        results = [self.corpus[idx] for idx in idxs[0]]
        if return_scores:
            return results, scores[0]
        return results

    def embed(self, query: str) -> np.ndarray:
        """
        Embeds the query using the OpenAI embedding model.
        Input:
            query: an input query to embed.
        Returns:
            The embedding of the query using the embedding model.
        """
        embedding = self.client.embeddings.create(
            input=query,
            model=self.model_name,
            dimensions=self.hidden_size
        )
        return np.array(embedding.data[0].embedding)
