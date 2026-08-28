"""
Context Relevance Reranker — Cross-Encoder based pre-NLI filter.

Problem being solved:
  RAG retrieval (dense/sparse) often returns chunks that are *topically related*
  to the query but contain no information about the specific claim being verified.
  Feeding irrelevant chunks into NLI causes false entailments (neutral chunks
  loosely matching claims) and inflates grounding scores incorrectly.

Solution:
  Before NLI verification, re-score every retrieved context chunk against the
  user's original query using a cross-encoder RERANKER model. Only chunks that
  score above ``relevance_threshold`` are passed to the NLI stage.

Models:
  cross-encoder/ms-marco-MiniLM-L-6-v2   (fast, 22MB, recommended)
  cross-encoder/ms-marco-electra-base     (higher accuracy, 135MB)
  cross-encoder/ms-marco-TinyBERT-L-2-v2 (ultra-fast, 4MB)

Output format:
  List[RankedChunk] — each chunk with its relevance score, sorted descending.

Latency:
  ≤10 chunks, MiniLM-L-6: ~5-15ms on CPU (single batch pass).

Pipeline position:
  [RAG Retrieved Chunks]
       ↓
  ContextReranker.rerank(query, chunks)   ← THIS MODULE
       ↓
  [Relevant Chunks Only (above threshold)]
       ↓
  NLIRAGGroundingGuard (claim-level NLI)
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("controlplane.detectors.hallucination.context_reranker")

# Shared thread pool — reranker inference must not block the event loop
_RERANKER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker_infer")

# Module-level singleton so the model is loaded exactly once
_SINGLETON_RERANKER: Optional[ContextReranker] = None


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RankedChunk:
    """A context chunk with its reranker relevance score."""
    text: str
    relevance_score: float   # [0, 1] after sigmoid (or raw logit if binary)
    original_index: int      # position in the original retrieved_context list


# ---------------------------------------------------------------------------
# ContextReranker
# ---------------------------------------------------------------------------

class ContextReranker:
    """Cross-encoder relevance reranker for RAG context chunks.

    Filters out context chunks that are unlikely to contain information
    relevant to the user's query, reducing NLI false positives.

    Args:
        model_name:           HuggingFace cross-encoder reranker model ID.
        relevance_threshold:  Minimum score to keep a chunk. Chunks below
                              this are dropped before NLI.
        top_k:                If set, keep at most ``top_k`` chunks regardless
                              of how many pass the threshold.
        max_length:           Tokenisation truncation limit per (query, chunk) pair.

    Usage::
        reranker = ContextReranker()
        ranked = reranker.rerank(
            query="What is the refund policy?",
            chunks=["Returns take 7 days.", "Our CEO joined in 2018.", ...],
        )
        relevant = [r.text for r in ranked]
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        relevance_threshold: float = 0.30,
        top_k: Optional[int] = None,
        max_length: int = 256,
    ) -> None:
        self._model_name = model_name
        self._relevance_threshold = relevance_threshold
        self._top_k = top_k
        self._max_length = max_length
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
            )
            # Warm-up with a dummy pair
            self._model.predict(
                [("warmup query", "warmup passage")],
                batch_size=1,
                show_progress_bar=False,
            )
            logger.info(
                "ContextReranker loaded: model=%s, threshold=%.2f, top_k=%s",
                self._model_name,
                self._relevance_threshold,
                self._top_k,
            )
        except Exception as exc:
            logger.error("ContextReranker failed to load '%s': %s", self._model_name, exc)
            raise

    # ------------------------------------------------------------------
    # Core reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: List[str],
    ) -> List[RankedChunk]:
        """Score all (query, chunk) pairs and return relevant chunks.

        Args:
            query:  The original user query (not the completion).
            chunks: Retrieved context chunks (the full list).

        Returns:
            List of ``RankedChunk`` sorted by ``relevance_score`` descending,
            filtered to only those ≥ ``relevance_threshold``.
            If no chunks pass the threshold, returns the single top-scoring
            chunk to prevent total context loss.
        """
        if not chunks or self._model is None:
            return [RankedChunk(text=c, relevance_score=1.0, original_index=i)
                    for i, c in enumerate(chunks)]

        pairs = [(query, chunk) for chunk in chunks]

        # ms-marco models output raw logits (not probabilities).
        # Apply sigmoid to map to [0,1] relevance probability.
        import numpy as np
        raw_scores = self._model.predict(
            pairs,
            batch_size=len(pairs),  # single pass for ≤typical RAG context sizes
            show_progress_bar=False,
        )
        # Sigmoid: handles both binary relevance and regression outputs
        relevance_probs = 1.0 / (1.0 + np.exp(-raw_scores))

        ranked: List[RankedChunk] = [
            RankedChunk(
                text=chunk,
                relevance_score=float(score),
                original_index=idx,
            )
            for idx, (chunk, score) in enumerate(zip(chunks, relevance_probs))
        ]

        # Sort descending by relevance
        ranked.sort(key=lambda r: r.relevance_score, reverse=True)

        # Apply threshold filter
        above_threshold = [r for r in ranked if r.relevance_score >= self._relevance_threshold]

        # Safety net: always keep at least the top chunk to avoid zero-context NLI
        filtered = above_threshold if above_threshold else [ranked[0]]

        # Apply top_k cap
        if self._top_k is not None:
            filtered = filtered[: self._top_k]

        logger.debug(
            "Reranker: %d/%d chunks kept (threshold=%.2f, top_k=%s). "
            "Top score=%.3f, bottom score=%.3f",
            len(filtered),
            len(chunks),
            self._relevance_threshold,
            self._top_k,
            filtered[0].relevance_score if filtered else 0.0,
            filtered[-1].relevance_score if filtered else 0.0,
        )
        return filtered

    async def async_rerank(
        self,
        query: str,
        chunks: List[str],
    ) -> List[RankedChunk]:
        """Async wrapper — offloads ``rerank`` to the shared thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _RERANKER_EXECUTOR,
            self.rerank,
            query,
            chunks,
        )


# ---------------------------------------------------------------------------
# Factory & singleton
# ---------------------------------------------------------------------------

def get_context_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    relevance_threshold: float = 0.30,
    top_k: Optional[int] = None,
) -> Optional[ContextReranker]:
    """Construct and cache a ``ContextReranker`` singleton.

    Returns ``None`` if ``sentence_transformers`` is not installed so the
    caller can skip reranking gracefully without crashing.
    """
    global _SINGLETON_RERANKER

    if _SINGLETON_RERANKER is not None:
        return _SINGLETON_RERANKER

    try:
        import sentence_transformers  # noqa: F401
        _SINGLETON_RERANKER = ContextReranker(
            model_name=model_name,
            relevance_threshold=relevance_threshold,
            top_k=top_k,
        )
        return _SINGLETON_RERANKER
    except Exception as exc:
        logger.warning(
            "ContextReranker unavailable (sentence_transformers not installed or model failed). "
            "Skipping context reranking. Error: %s",
            exc,
        )
        return None
