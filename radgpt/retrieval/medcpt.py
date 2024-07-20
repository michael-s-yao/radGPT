"""
MedCPT- based retriever class implementation.

Author(s):
    Michael Yao @michael-s-yao
    Allison Chae @allisonjchae

Citation(s):
    [1] Jin Q, Kim W, Chen Q, Comeau DC, Yeganova L, Wilbur WJ, Lu Z. MedCPT:
        Contrastive pre-trained transformers with large-scale PubMed search
        logs for zero-shot biomedical information retrieval. Bioinform 39(11):
        btad651. (2023). doi: 10.1093/bioinformatics/btad651

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import faiss
import os
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .base import Corpus, Document, Retriever


class MedCPTRetriever(Retriever):
    model_name: str = "ncbi/MedCPT-Query-Encoder"

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
        super(MedCPTRetriever, self).__init__(
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
        Embeds the query using the MedCPT model.
        Input:
            query: an input query.
        Returns:
            The embedding of the query using the MedCPT model.
        """
        kwargs = {
            key: val[:, :min(self.max_tokens, val.size(dim=-1))]
            for key, val in self.tokenizer(query, return_tensors="pt").items()
        }
        embedding = self.model(**kwargs)
        out = embedding.last_hidden_state[:, 0, :]
        return out / torch.linalg.norm(out, dim=-1)
