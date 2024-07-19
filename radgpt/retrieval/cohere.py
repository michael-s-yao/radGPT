"""
Cohere English embedding model-based retriever.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import cohere_aws
import faiss
import numpy as np
import os
from configparser import RawConfigParser
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .base import Corpus, Document, Retriever


class CohereRetriever(Retriever):
    model_name: str = "cohere.embed-english-v3"

    hidden_size: int = 1024

    batch_size: int = 16

    def __init__(
        self,
        corpus: Corpus,
        index_dir: Optional[Union[Path, str]] = os.path.join(
            os.path.dirname(__file__), "indices"
        ),
        credentials_fn: Union[Path, str] = os.path.expanduser(
            "~/.aws/credentials"
        ),
        profile: str = "default",
        **kwargs
    ):
        """
        Args:
            corpus: a corpus of documents to retrieve from.
            index_dir: the cache path to load and save the index from.
            credentials_fn: the filename to the AWS credentials.
            profile: the credentials profile to use. Default `default`.
        """
        super(CohereRetriever, self).__init__(
            corpus=corpus, index_dir=index_dir, **kwargs
        )
        config = RawConfigParser()
        config.read(credentials_fn)
        self.client = cohere_aws.Client(
            mode=cohere_aws.Mode.BEDROCK, region_name="us-east-1"
        )

        if os.path.isfile(self.index_fn):
            self.index = faiss.read_index(self.index_fn)
            return
        self.index = faiss.IndexFlatIP(self.hidden_size)
        embeddings = self.embed([doc.text for doc in self.corpus])
        for vec in embeddings:
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
        embedding = self.embed(query)
        results = self.index.search(embedding, k=k)
        scores, idxs = self.index.search(embedding, k=k)
        results = [self.corpus[idx] for idx in idxs[0]]
        if return_scores:
            return results, scores[0]
        return results

    def embed(self, query: Union[str, Sequence[str]]) -> np.ndarray:
        """
        Embeds the query using the Cohere embedding model.
        Input:
            query: an input query or list of queries.
        Returns:
            The embedding of the query using the embedding model.
        """
        query = [query] if isinstance(query, str) else query

        embeddings = []
        for i in range(0, len(query), self.batch_size):
            embed = self.client.embed(
                texts=query[i:min(i + self.batch_size, len(query))],
                model_id=self.model_name,
                input_type="search_document",
            )
            embeddings.append(np.array(embed.embeddings))
        return np.vstack(embeddings)
