"""
Base retriever class implementation.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from __future__ import annotations
import abc
import numpy as np
import os
from datasets import load_dataset
from pathlib import Path
from typing import Generator, Optional, Sequence, Type, Union


class Document:
    def __init__(self, uid: str, text: str):
        """
        Args:
            uid: the unique document identifier.
            text: the text of the document.
        """
        self.uid = uid
        self.text = text

    def __len__(self) -> int:
        """
        Returns the number of characters in the document.
        Input:
            None.
        Returns:
            The number of characters in the document.
        """
        return len(self.text)

    def __str__(self) -> str:
        """
        Returns a string representation of the document text.
        Input:
            None.
        Returns:
            A string representation of the document text.
        """
        return self.text

    def __repr__(self) -> str:
        """
        Returns a string representation of the document text.
        Input:
            None.
        Returns:
            A string representation of the document text.
        """
        return str(self)

    def lower(self) -> str:
        """
        Returns a lower case string representation of the document text.
        Input:
            None.
        Returns:
            A lower case string representation of the document text.
        """
        return self.text.lower()


class Corpus:
    def __init__(
        self, name: Optional[str] = None, dataset: Optional[np.ndarray] = None
    ):
        """
        Args:
            name: the name of the corpus dataset to load if initializing from
                a named dataset.
            dataset: the corpus dataset if initializing from an array.
        """
        self.name = name
        self.dataset = dataset
        assert bool(self.name is None) != bool(self.dataset is None), (
            "Only one of `name`, `dataset` can be specified."
        )
        if self.name is not None:
            ds = load_dataset(self.name)["train"]
            self._corpus = list(
                map(
                    lambda doc: Document(uid=doc["id"], text=doc["contents"]),
                    ds
                )
            )
        else:
            self._corpus = [
                Document(uid=str(idx), text=text)
                for idx, text in enumerate(dataset)
            ]

    @property
    def uid(self) -> Sequence[str]:
        """
        Returns a list of the document UIDs.
        Input:
            None.
        Returns:
            A list of the document UIDs.
        """
        return list(map(lambda doc: doc.uid, self._corpus))

    def __len__(self) -> int:
        """
        Returns the number of documents in the corpus.
        Input:
            None.
        Returns:
            The number of documents in the corpus.
        """
        return len(self._corpus)

    def __iter__(self) -> Generator:
        """
        Returns an iterable over the corpus of documents.
        Input:
            None.
        Returns:
            An iterable over the corpus of documents.
        """
        yield from self._corpus

    def __getitem__(self, idx: Union[str, int]) -> Document:
        """
        Returns a specified document from the corpus.
        Input:
            idx: the index of the document in the corpus. Can be either a UID
                string or an integer position in the corpus list.
        Returns:
            The specified document from the corpus.
        """
        if isinstance(idx, str):
            return next(filter(lambda doc: doc.uid == idx, self._corpus))
        return self._corpus[idx]


class Retriever(abc.ABC):
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
        self.corpus = corpus
        if index_dir is not None:
            self.index_fn = os.path.join(
                index_dir, f"{corpus.name}_{self.__class__.__name__}.index"
            )
            os.makedirs(os.path.dirname(self.index_fn), exist_ok=True)
        for key, val in kwargs.items():
            setattr(self, key, val)

    @abc.abstractmethod
    def retrieve(self, query: str, k: int, **kwargs) -> Sequence[Document]:
        """
        Retrieves the top k documents relevant to the query.
        Input:
            query: an input query.
            k: the number of documents to retrieve.
        Retrieve:
            The top k documents relevant to the query.
        """
        raise NotImplementedError

    @classmethod
    def from_dataset(
        cls: Type[Retriever], dataset: np.ndarray, **kwargs
    ) -> Retriever:
        """
        Builds a retriever from a pre-existing dataset of text data.
        Input:
            cls: the retriever class.
            dataset: the dataset to build the retriever from.
        Returns:
            The instantiated retriever from the text dataset.
        """
        return cls(corpus=Corpus(dataset=dataset), **kwargs)


class RandomRetriever(Retriever):
    def __init__(self, corpus: Corpus, seed: int = 42, **kwargs):
        """
        Args:
            corpus: a corpus of documents to retrieve from.
            seed: random seed. Default 42.
        """
        super(RandomRetriever, self).__init__(
            corpus, index_dir=None, seed=seed, **kwargs
        )
        self._rng = np.random.default_rng(seed=self.seed)

    def retrieve(self, query: str, k: int, **kwargs) -> Sequence[Document]:
        """
        Retrieves a random set of k documents.
        Input:
            query: an input query.
            k: the number of documents to retrieve.
        Retrieve:
            A list of k random documents from the corpus.
        """
        return [
            self.corpus[i]
            for i in self._rng.choice(len(self.corpus), size=k, replace=False)
        ]
