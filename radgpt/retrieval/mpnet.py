"""
MPNet- based retriever class implementations.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Citation(s):
    [1] Song K, Tan X, Qin T, Lu J. Li T. MPNet: Masked and permuted pre-
        training for language understanding. arXiv Preprint. (2020). doi:
        10.48550/arXiv.2004.09297

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import faiss
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .base import Corpus, Document, Retriever


class MPNetRetriever(Retriever):
    model_name: str = "sentence-transformers/all-mpnet-base-v2"

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
        super(MPNetRetriever, self).__init__(
            corpus=corpus, index_dir=index_dir, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        *_, last = self.model.parameters()
        hidden_size = last.size(dim=-1)

        if os.path.isfile(self.index_fn):
            self.index = faiss.read_index(self.index_fn)
            return
        self.index = faiss.IndexFlatIP(hidden_size)
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
        tokens = self.tokenizer(query, return_tensors="pt", truncation=True)
        output = self.model(**tokens)
        mask = tokens["attention_mask"].unsqueeze(dim=-1).expand(
            output[0].size()
        )
        mask = mask.float()
        embedding = torch.sum(output[0] * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=torch.finfo(torch.float32).eps
        )
        return F.normalize(embedding, p=2, dim=1)
