"""
Base retriever class implementation.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import abc
from datasets import load_dataset
from typing import Generator, Sequence, Union


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
    def __init__(self, name: str):
        """
        Args:
            name: the name of the corpus dataset to load.
        """
        self.name = name
        ds = load_dataset(self.name)["train"]
        self._corpus = list(
            map(lambda doc: Document(uid=doc["id"], text=doc["contents"]), ds)
        )

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
    def __init__(self, corpus: Corpus, **kwargs):
        """
        Args:
            corpus: a corpus of documents to retrieve from.
        """
        self.corpus = corpus
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
