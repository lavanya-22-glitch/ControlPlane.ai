"""
rag/reranker.py
───────────────
Pure-numpy cosine similarity re-ranker — completely torch-free.

Instead of a cross-encoder (which requires PyTorch), we re-embed both the query
and each candidate chunk using the same ONNX MiniLM embedder, then rank by
cosine similarity. Since embeddings are L2-normalised, cosine sim = dot product.

This is lighter than a full cross-encoder but significantly more precise than
the initial hybrid RRF pool because it gives a focused per-pair score.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from langchain_core.documents import Document

from rag.config import RERANK_TOP_N
from rag.embedder import _embed_batch, _load_models


class CosineReranker:
    """
    Re-ranks a list of Documents against a query using cosine similarity
    between ONNX-computed embeddings (no torch required).
    """

    def __init__(self) -> None:
        # Warm up the ONNX model so first query isn't slow
        _load_models()
        print("[Reranker] Cosine similarity reranker ready (ONNX-backed, torch-free).")

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_n: int = RERANK_TOP_N,
    ) -> List[Tuple[Document, float]]:
        """
        Returns top_n (Document, cosine_score) tuples sorted descending.
        """
        if not docs:
            return []

        # Embed query + all docs in one batched call
        texts = [query] + [d.page_content for d in docs]
        vecs  = _embed_batch(texts)   # shape: (1 + len(docs), dim), L2-normalised

        q_vec   = vecs[0]             # (dim,)
        doc_vecs = vecs[1:]           # (n_docs, dim)

        # Cosine similarity = dot product (since vectors are unit-norm)
        scores: List[float] = (doc_vecs @ q_vec).tolist()

        scored = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_n]


# Keep backward-compat alias in case anything imports CrossEncoderReranker
CrossEncoderReranker = CosineReranker

