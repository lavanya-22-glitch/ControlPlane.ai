"""
rag/hybrid_retriever.py
────────────────────────
Hybrid retrieval: dense FAISS + sparse BM25 merged via Reciprocal Rank Fusion (RRF).

RRF formula: score(d) = Σ 1 / (k + rank(d))
where k=60 is a smoothing constant that reduces the impact of very high-rank results.
This is parameter-free, robust, and consistently outperforms score-based fusion.
"""

from __future__ import annotations

from typing import List, Dict, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.bm25_retriever import BM25Retriever
from rag.config import DENSE_TOP_K, SPARSE_TOP_K

RRF_K = 60  # standard RRF smoothing constant


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)  # +1 because rank is 0-indexed


class HybridRetriever:
    """
    Combines FAISS dense and BM25 sparse retrieval via Reciprocal Rank Fusion.
    Returns a merged, deduplicated list of Documents sorted by fused RRF score.
    """

    def __init__(self, faiss_store: FAISS, bm25: BM25Retriever) -> None:
        self._faiss = faiss_store
        self._bm25 = bm25

    def retrieve(
        self,
        query: str,
        dense_top_k: int = DENSE_TOP_K,
        sparse_top_k: int = SPARSE_TOP_K,
    ) -> List[Tuple[Document, float]]:
        """
        Returns list of (Document, rrf_score) sorted descending by fused score.
        """
        # ── Dense retrieval (FAISS cosine similarity) ──────────────────────
        dense_results: List[Tuple[Document, float]] = (
            self._faiss.similarity_search_with_score(query, k=dense_top_k)
        )

        # ── Sparse retrieval (BM25) ────────────────────────────────────────
        sparse_results: List[Tuple[Document, float]] = self._bm25.retrieve(
            query, top_k=sparse_top_k
        )

        # ── RRF fusion ─────────────────────────────────────────────────────
        # Use page_content as dedup key (chunk IDs may differ across stores)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, (doc, _score) in enumerate(dense_results):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + _rrf_score(rank)
            doc_map[key] = doc

        for rank, (doc, _score) in enumerate(sparse_results):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + _rrf_score(rank)
            doc_map[key] = doc

        # Sort by descending RRF score
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        return [(doc_map[k], rrf_scores[k]) for k in sorted_keys]
