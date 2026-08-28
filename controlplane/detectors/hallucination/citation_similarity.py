"""
Citation Similarity Checker — RAG grounding via citation-aware similarity scoring.

Problem:
  Standard NLI checks score EVERY sentence against context, but citations are the
  LLM's explicit claim that a specific sentence is backed by a specific source.
  A citation with low similarity to ANY context chunk is a strong hallucination signal.

Two-tier similarity scoring (no sklearn required):
  Tier 1 — N-gram overlap (always available, <1ms):
    Bigram Dice coefficient: 2 * |bigrams(a) ∩ bigrams(b)| / (|bigrams(a)| + |bigrams(b)|)
    Combined with word unigram recall against context.
    Final score = 0.5 * dice + 0.5 * word_recall

  Tier 2 — Sentence embedding cosine (optional, requires sentence-transformers):
    Bi-encoder: encode cited_text and context_chunk independently, then cosine similarity.
    Faster than cross-encoder for this use-case since we need ALL-pairs scoring.

Short-circuit behaviour:
  As soon as any citation scores below ``fail_threshold``, the detector returns
  FAIL immediately — no remaining citations are evaluated. This is the per-citation
  short circuit requested.

Pipeline position (runs AFTER NLI in router.py):
  [LLM Completion]
       ↓
  CitationParser.parse()          ← extract [1], [Doc A], etc.
       ↓
  For each Citation:
    find_best_chunk()             ← best matching context chunk (n-gram similarity)
    score_similarity()            ← Tier 1 or Tier 2 similarity
    score < fail_threshold?  ──► SHORT CIRCUIT → FAIL (REWRITE/BLOCK)
       ↓
  avg(all citation scores) < avg_threshold?  ──► FAIL
       ↓
  PASS (with full citation metadata in audit trace)
"""
from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from controlplane.detectors.base import BaseDetector, DetectionResult, GuardContext, GuardStage
from controlplane.detectors.hallucination.citation_parser import Citation, CitationParser
from controlplane.pdp.types import PDPAction, ViolationCode

logger = logging.getLogger("controlplane.detectors.hallucination.citation_similarity")

_CITATION_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="citation_sim")

# Stopwords for word-level recall (minimal set, no external data file)
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "and", "or", "but", "if", "as",
    "this", "that", "it", "its", "their", "our", "we", "they", "he",
    "she", "i", "you", "from", "into", "not", "no", "any", "all",
})


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CitationSimilarityResult:
    """Similarity check result for a single citation."""
    citation: Citation
    best_chunk: str           # The context chunk most similar to the cited text
    best_chunk_index: int     # Index in retrieved_context list
    similarity_score: float   # [0, 1] — higher is more grounded
    is_grounded: bool         # similarity_score >= fail_threshold
    scorer_name: str          # "ngram" | "embedding"

    def to_dict(self) -> dict:
        return {
            "marker": self.citation.marker,
            "marker_type": self.citation.marker_type,
            "preceding_text": self.citation.preceding_text[:120],
            "best_chunk": self.best_chunk[:120],
            "similarity_score": round(self.similarity_score, 3),
            "is_grounded": self.is_grounded,
            "scorer": self.scorer_name,
        }


# ---------------------------------------------------------------------------
# Similarity scorer ABC + implementations
# ---------------------------------------------------------------------------

class CitationSimilarityScorer(ABC):
    """Abstract similarity scorer for (cited_text, context_chunk) pairs."""

    @property
    @abstractmethod
    def scorer_name(self) -> str: ...

    @abstractmethod
    def score(self, text_a: str, text_b: str) -> float:
        """Return similarity score in [0, 1]."""
        ...

    def score_all_chunks(self, cited_text: str, chunks: List[str]) -> Tuple[str, int, float]:
        """Score cited_text against all chunks and return (best_chunk, index, score)."""
        if not chunks:
            return ("", -1, 0.0)
        scores = [self.score(cited_text, chunk) for chunk in chunks]
        best_idx = int(np.argmax(scores))
        return chunks[best_idx], best_idx, float(scores[best_idx])


class NgramSimilarityScorer(CitationSimilarityScorer):
    """Bigram Dice coefficient + content-word recall.

    Combined score = 0.6 * bigram_dice + 0.4 * word_recall
    No external dependencies — runs in microseconds per pair.

    Bigram Dice: captures phrase-level overlap (surface form similarity).
    Word recall: measures how many content words from the citation appear
                 in the context chunk, penalising stopword-heavy matches.
    """

    @property
    def scorer_name(self) -> str:
        return "ngram"

    @staticmethod
    def _bigrams(text: str) -> set:
        """Character-level bigrams of lowercased text."""
        t = text.lower()
        return {t[i:i+2] for i in range(len(t) - 1)} if len(t) >= 2 else set()

    @staticmethod
    def _content_words(text: str) -> frozenset:
        words = re.findall(r"\w+", text.lower())
        return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)

    def score(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0

        # Bigram Dice coefficient
        bg_a = self._bigrams(text_a)
        bg_b = self._bigrams(text_b)
        if bg_a or bg_b:
            dice = 2 * len(bg_a & bg_b) / (len(bg_a) + len(bg_b))
        else:
            dice = 0.0

        # Content-word recall: what fraction of citation's content words appear in chunk?
        words_a = self._content_words(text_a)
        words_b = self._content_words(text_b)
        if words_a:
            word_recall = len(words_a & words_b) / len(words_a)
        else:
            word_recall = 0.0

        return 0.6 * dice + 0.4 * word_recall


class EmbeddingSimilarityScorer(CitationSimilarityScorer):
    """Sentence embedding cosine similarity using sentence-transformers bi-encoder.

    Uses a bi-encoder (not a cross-encoder) for citation similarity because we
    need to compare ONE cited text against MANY context chunks — encoding chunks
    once at construction time would be ideal, but here we encode both texts per call.

    Latency: ~5-20ms per pair (MiniLM-L6, CPU).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            # Warm-up
            self._model.encode(["warmup"], show_progress_bar=False)
            logger.info("EmbeddingSimilarityScorer loaded: model=%s", self._model_name)
        except Exception as exc:
            logger.error("EmbeddingSimilarityScorer failed to load: %s", exc)
            raise

    @property
    def scorer_name(self) -> str:
        return f"embedding:{self._model_name}"

    def score(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b or self._model is None:
            return 0.0
        embeddings = self._model.encode(
            [text_a, text_b],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        # Cosine similarity = dot product (vectors are already L2-normalised)
        return float(np.dot(embeddings[0], embeddings[1]))

    def score_all_chunks(self, cited_text: str, chunks: List[str]) -> Tuple[str, int, float]:
        """Optimised: encode cited_text + all chunks in one batch."""
        if not chunks or self._model is None:
            return ("", -1, 0.0)
        all_texts = [cited_text] + chunks
        embeddings = self._model.encode(
            all_texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        cited_emb = embeddings[0]
        chunk_embs = embeddings[1:]
        scores = chunk_embs @ cited_emb  # dot products, shape (N,)
        best_idx = int(np.argmax(scores))
        return chunks[best_idx], best_idx, float(scores[best_idx])


def _build_scorer(method: str = "auto") -> CitationSimilarityScorer:
    """Factory: return the best available scorer.

    Priority (when ``method="auto"``):
      1. EmbeddingSimilarityScorer — if sentence-transformers is installed
      2. NgramSimilarityScorer    — always available
    """
    if method == "embedding":
        return EmbeddingSimilarityScorer()

    if method == "ngram":
        return NgramSimilarityScorer()

    # auto: try embedding first, fall back to ngram
    try:
        scorer = EmbeddingSimilarityScorer()
        logger.info("Citation similarity scorer: embedding (all-MiniLM-L6-v2)")
        return scorer
    except Exception as exc:
        logger.info(
            "Embedding scorer unavailable (%s); using n-gram scorer.", exc
        )
        return NgramSimilarityScorer()


# ---------------------------------------------------------------------------
# CitationGroundingGuard — BaseDetector implementation
# ---------------------------------------------------------------------------

class CitationGroundingGuard(BaseDetector):
    """Citation-aware grounding detector for RAG completions.

    For each citation marker found in the completion ([1], [Doc A], etc.),
    scores the similarity between the preceding claimed text and the best
    matching context chunk.

    Short-circuit: aborts immediately when any citation falls below
    ``citation_fail_threshold`` — the remaining citations are not scored.

    Args:
        scorer: Explicit ``CitationSimilarityScorer``. If ``None``,
                ``_build_scorer("auto")`` is called (lazy, cached as singleton).

    Usage::
        guard = CitationGroundingGuard()
        result = await guard.evaluate(ctx)
    """

    _scorer_singleton: Optional[CitationSimilarityScorer] = None

    def __init__(self, scorer: Optional[CitationSimilarityScorer] = None) -> None:
        self._scorer = scorer
        self._parser = CitationParser()

    def _get_scorer(self, method: str = "auto") -> CitationSimilarityScorer:
        if self._scorer is not None:
            return self._scorer
        # Lazy singleton construction
        if CitationGroundingGuard._scorer_singleton is None:
            CitationGroundingGuard._scorer_singleton = _build_scorer(method)
        return CitationGroundingGuard._scorer_singleton

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "citation_grounding_guard"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.POST_EXECUTION

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        """Run citation similarity check on ctx.completion_text."""
        hallu_config = (
            ctx.policy.post_execution.hallucination_engine
            if ctx.policy.post_execution
            else None
        )

        # Short-circuit: citation check not configured or disabled
        citation_enabled = getattr(hallu_config, "citation_check_enabled", False)
        if hallu_config is None or not hallu_config.enabled or not citation_enabled:
            return self._allow(score=None, metadata={"reason": "citation_check_disabled"})

        context_chunks: List[str] = ctx.retrieved_context or []
        completion: str = ctx.completion_text or ""

        if not context_chunks or not completion:
            return self._allow(score=None, metadata={"reason": "no_context_or_completion"})

        # ---- Step 1: Parse citation markers --------------------------------
        citations: List[Citation] = self._parser.parse(completion)

        if not citations:
            return self._allow(
                score=None,
                metadata={"reason": "no_citations_found", "has_citations": False},
            )

        logger.debug(
            "Trace %s: Found %d citation(s) in completion. Running similarity check.",
            ctx.trace_id,
            len(citations),
        )

        # ---- Step 2: Score each citation (sync, offloaded to executor) -----
        fail_threshold: float = getattr(hallu_config, "citation_fail_threshold", 0.35)
        avg_threshold: float = getattr(hallu_config, "citation_avg_threshold", 0.50)
        similarity_method: str = getattr(hallu_config, "citation_similarity_method", "auto")

        loop = asyncio.get_event_loop()
        results: List[CitationSimilarityResult] = await loop.run_in_executor(
            _CITATION_EXECUTOR,
            self._score_citations,
            citations,
            context_chunks,
            fail_threshold,
            similarity_method,
        )

        # ---- Step 3: Short-circuit on first ungrounded citation -----------
        for res in results:
            if not res.is_grounded:
                reason = (
                    f"Citation short-circuit: {res.citation.marker!r} preceding text "
                    f"has similarity {res.similarity_score:.3f} < fail_threshold "
                    f"{fail_threshold:.2f}. Claimed: \"{res.citation.preceding_text[:80]}...\""
                )
                logger.warning("Trace %s: %s", ctx.trace_id, reason)
                return self._fail(
                    score=res.similarity_score,
                    hallu_config=hallu_config,
                    reason=reason,
                    metadata={
                        "citations_total": len(citations),
                        "citations_evaluated": results.index(res) + 1,
                        "short_circuit": True,
                        "failed_citation": res.to_dict(),
                        "all_citation_scores": [r.to_dict() for r in results],
                        "scorer": res.scorer_name,
                    },
                )

        # ---- Step 4: Aggregate threshold check ----------------------------
        scores = [r.similarity_score for r in results]
        avg_score = float(np.mean(scores)) if scores else 1.0
        passed = avg_score >= avg_threshold

        metadata = {
            "citations_total": len(citations),
            "citations_evaluated": len(results),
            "avg_similarity_score": round(avg_score, 3),
            "short_circuit": False,
            "has_citations": True,
            "all_citation_scores": [r.to_dict() for r in results],
            "scorer": results[0].scorer_name if results else "none",
        }

        if not passed:
            reason = (
                f"Citation average similarity {avg_score:.3f} < avg_threshold {avg_threshold:.2f} "
                f"across {len(citations)} citation(s)."
            )
            logger.warning("Trace %s: %s", ctx.trace_id, reason)
            return self._fail(
                score=avg_score,
                hallu_config=hallu_config,
                reason=reason,
                metadata=metadata,
            )

        logger.info(
            "Trace %s: Citation check PASSED. avg_similarity=%.3f across %d citation(s).",
            ctx.trace_id,
            avg_score,
            len(citations),
        )
        return self._allow(score=avg_score, metadata=metadata)

    # ------------------------------------------------------------------
    # Synchronous scoring (runs in thread pool)
    # ------------------------------------------------------------------

    def _score_citations(
        self,
        citations: List[Citation],
        context_chunks: List[str],
        fail_threshold: float,
        method: str,
    ) -> List[CitationSimilarityResult]:
        """Score all citations synchronously.

        Called inside the thread-pool executor — must not use asyncio.
        Short-circuits after the first failing citation to save time.
        """
        scorer = self._get_scorer(method)
        results: List[CitationSimilarityResult] = []

        for citation in citations:
            best_chunk, best_idx, sim_score = scorer.score_all_chunks(
                cited_text=citation.preceding_text,
                chunks=context_chunks,
            )
            is_grounded = sim_score >= fail_threshold
            results.append(CitationSimilarityResult(
                citation=citation,
                best_chunk=best_chunk,
                best_chunk_index=best_idx,
                similarity_score=sim_score,
                is_grounded=is_grounded,
                scorer_name=scorer.scorer_name,
            ))
            # Short-circuit: stop scoring if we hit a failing citation
            if not is_grounded:
                logger.debug(
                    "Citation short-circuit in _score_citations: marker=%s score=%.3f",
                    citation.marker,
                    sim_score,
                )
                break

        return results

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _allow(self, score: Optional[float], metadata: dict) -> DetectionResult:
        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            score=score,
            suggested_action=PDPAction.ALLOW,
            metadata=metadata,
        )

    def _fail(self, score: float, hallu_config, reason: str, metadata: dict) -> DetectionResult:
        action = (
            PDPAction.REWRITE
            if hallu_config.action_on_low_grounding == "rewrite"
            else PDPAction.BLOCK
        )
        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=False,
            score=score,
            suggested_action=action,
            violation_code=ViolationCode.HALLUCINATION,
            reason=reason,
            metadata=metadata,
        )
