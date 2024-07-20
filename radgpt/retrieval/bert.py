"""
BERT- and RadBERT- based retriever class implementations.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Citation(s):
    [1] Yan A, McAuley J, Lu X, Du J, Chang EY, Gentili A, Hsu C. RadBERT:
        Adapting transformer-based language models to radiology. Radiology:
        Artificial Intelligence 4(4): e210258. (2022). doi: 10.1148/ryai.210258

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import faiss
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .base import Corpus, Document, Retriever


class BERTRetriever(Retriever):
    model_name: str = "FacebookAI/roberta-base"

    max_tokens: int = 512

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
        super(BERTRetriever, self).__init__(
            corpus=corpus, index_dir=index_dir, **kwargs
        )
        config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, config=config)

        if os.path.isfile(self.index_fn):
            self.index = faiss.read_index(self.index_fn)
            return
        self.index = faiss.IndexFlatIP(config.hidden_size)
        embeddings = map(lambda doc: self.embed(doc.text), self.corpus)
        for vec in embeddings:
            self.index.add(vec.detach().cpu().numpy())
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
        embedding = self.embed(query).detach().cpu().numpy()
        results = self.index.search(embedding, k=k)
        scores, idxs = self.index.search(embedding, k=k)
        results = [self.corpus[idx] for idx in idxs[0]]
        if return_scores:
            return results, scores[0]
        return results

    @torch.no_grad()
    def embed(self, query: str) -> torch.Tensor:
        """
        Embeds the query using the BERT model.
        Input:
            query: an input query.
        Returns:
            The embedding of the query using the BERT model.
        """
        kwargs = {
            key: val[:, :min(self.max_tokens, val.size(dim=-1))]
            for key, val in self.tokenizer(query, return_tensors="pt").items()
        }
        embedding = self.model(**kwargs)
        out = embedding.pooler_output
        return out / torch.linalg.norm(out, dim=-1)


class RadBERTRetriever(BERTRetriever):
    model_name: str = "zzxslp/RadBERT-RoBERTa-4m"
