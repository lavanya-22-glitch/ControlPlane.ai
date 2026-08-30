"""
rag/bm25_retriever.py
──────────────────────
Sparse BM25 retrieval over the chunked corpus.
Uses rank_bm25 (BM25Okapi) — pure Python, zero extra deps.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from rag.config import SPARSE_TOP_K


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer."""
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class BM25Retriever:
    """BM25 sparse retriever over a fixed document corpus."""

    def __init__(self, docs: List[Document]) -> None:
        self._docs = docs
        tokenized = [_tokenize(d.page_content) for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        print(f"[BM25] Index built over {len(docs)} documents.")

    def retrieve(
        self, query: str, top_k: int = SPARSE_TOP_K
    ) -> List[Tuple[Document, float]]:
        """
        Return top_k (document, score) tuples sorted by BM25 score descending.
        """
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair each doc with its score and sort
        scored = sorted(
            zip(self._docs, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]
