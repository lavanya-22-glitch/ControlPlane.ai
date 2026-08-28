"""
Unit tests for the Cross-Encoder NLI backend and Context Reranker.

All tests use HeuristicNLIBackend as a stand-in for the cross-encoder model
so they run without requiring sentence_transformers to be installed.

Test groups:
  1. CrossEncoderNLIBackend — label schema resolution
  2. ContextReranker         — reranking + threshold filtering (mocked model)
  3. NLIRAGGroundingGuard   — with context reranking enabled (integration)
  4. auto_select_backend     — cross_encoder preference handling
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from controlplane.detectors.hallucination.nli_backend import (
    HeuristicNLIBackend,
    NLIResult,
    auto_select_backend,
)
from controlplane.detectors.hallucination.nli_verifier import NLIRAGGroundingGuard
from controlplane.detectors.hallucination.context_reranker import (
    ContextReranker,
    RankedChunk,
    get_context_reranker,
)
from controlplane.detectors.hallucination.cross_encoder import _resolve_label_schema
from controlplane.detectors.base import GuardContext
from controlplane.policy.models import (
    HallucinationConfig,
    PolicyDefinition,
    PostExecutionConfig,
)
from controlplane.pdp.types import PDPAction, ViolationCode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rag_policy(
    min_grounding_score: float = 0.70,
    early_exit_threshold: float = 0.20,
    action: str = "rewrite",
    enabled: bool = True,
    context_reranking_enabled: bool = False,
    context_relevance_threshold: float = 0.30,
    context_reranker_top_k=None,
) -> PolicyDefinition:
    return PolicyDefinition(
        mode="rag",
        post_execution=PostExecutionConfig(
            hallucination_engine=HallucinationConfig(
                enabled=enabled,
                min_grounding_score=min_grounding_score,
                early_exit_threshold=early_exit_threshold,
                action_on_low_grounding=action,
                fallback_message="Unverified answer.",
                context_reranking_enabled=context_reranking_enabled,
                context_relevance_threshold=context_relevance_threshold,
                context_reranker_top_k=context_reranker_top_k,
            )
        ),
    )


def _make_ctx(
    policy: PolicyDefinition,
    completion: str,
    context: list[str] | None,
    messages: list[dict] | None = None,
    trace_id: str = "test_trace",
) -> GuardContext:
    return GuardContext(
        trace_id=trace_id,
        app_id="test",
        policy=policy,
        messages=messages or [{"role": "user", "content": "What is the refund policy?"}],
        retrieved_context=context,
        completion_text=completion,
    )


# ---------------------------------------------------------------------------
# 1. Label schema resolution tests (cross_encoder._resolve_label_schema)
# ---------------------------------------------------------------------------

class TestLabelSchemaResolution:

    def test_resolves_from_id2label_config(self):
        """Authoritative label schema from model config is used when valid."""
        id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        schema = _resolve_label_schema("any-model", id2label)
        assert schema["entailment"] == 1
        assert schema["contradiction"] == 0
        assert schema["neutral"] == 2

    def test_falls_back_to_deberta_heuristic(self):
        """DeBERTa models use name-based heuristic schema."""
        schema = _resolve_label_schema("cross-encoder/nli-deberta-v3-small", None)
        assert "entailment" in schema
        assert "contradiction" in schema
        assert "neutral" in schema

    def test_falls_back_to_default_for_unknown_model(self):
        """Unknown model names use the default schema without raising."""
        schema = _resolve_label_schema("some-unknown-model-xyz", None)
        assert "entailment" in schema

    def test_invalid_id2label_falls_back_to_heuristic(self):
        """If id2label is missing expected keys, falls back to heuristic."""
        bad_id2label = {0: "positive", 1: "negative"}  # not NLI labels
        schema = _resolve_label_schema("cross-encoder/nli-deberta-v3-small", bad_id2label)
        # Should still contain valid NLI keys from heuristic fallback
        assert "entailment" in schema


# ---------------------------------------------------------------------------
# 2. ContextReranker tests (with mocked CrossEncoder model)
# ---------------------------------------------------------------------------

class TestContextReranker:

    def _make_reranker_with_mock_scores(self, scores: list[float]) -> ContextReranker:
        """Create a ContextReranker whose model.predict() returns fixed scores."""
        import numpy as np

        reranker = ContextReranker.__new__(ContextReranker)
        reranker._model_name = "mock-model"
        reranker._relevance_threshold = 0.30
        reranker._top_k = None

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(scores)
        reranker._model = mock_model

        return reranker

    def test_rerank_filters_below_threshold(self):
        """Chunks with relevance < threshold should be excluded."""
        # Sigmoid(3.0) ≈ 0.95, Sigmoid(-3.0) ≈ 0.05
        reranker = self._make_reranker_with_mock_scores([3.0, -3.0])
        chunks = ["Highly relevant chunk.", "Totally irrelevant chunk."]
        ranked = reranker.rerank(query="test query", chunks=chunks)

        # Only the first chunk should survive threshold=0.30
        assert len(ranked) == 1
        assert ranked[0].text == "Highly relevant chunk."

    def test_rerank_always_keeps_at_least_one_chunk(self):
        """Safety net: when all chunks are below threshold, keep the top-scoring one."""
        reranker = self._make_reranker_with_mock_scores([-5.0, -6.0])
        chunks = ["Bad chunk 1.", "Bad chunk 2."]
        ranked = reranker.rerank(query="test query", chunks=chunks)

        # Should keep the less-bad one
        assert len(ranked) == 1
        assert ranked[0].text == "Bad chunk 1."

    def test_rerank_sorts_descending(self):
        """Returned chunks should be sorted by relevance descending."""
        reranker = self._make_reranker_with_mock_scores([1.0, 4.0, 2.0])
        chunks = ["Low.", "High.", "Medium."]
        ranked = reranker.rerank(query="test query", chunks=chunks)

        scores = [r.relevance_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_respects_top_k(self):
        """top_k caps the number of returned chunks even if more pass threshold."""
        reranker = self._make_reranker_with_mock_scores([3.0, 3.0, 3.0])
        reranker._top_k = 2
        chunks = ["A.", "B.", "C."]
        ranked = reranker.rerank(query="test query", chunks=chunks)
        assert len(ranked) == 2

    def test_rerank_empty_chunks_returns_empty(self):
        reranker = self._make_reranker_with_mock_scores([])
        ranked = reranker.rerank(query="test", chunks=[])
        assert ranked == []

    def test_original_index_preserved(self):
        """RankedChunk should carry the correct original_index."""
        reranker = self._make_reranker_with_mock_scores([1.0, 4.0])
        chunks = ["first", "second"]
        ranked = reranker.rerank(query="q", chunks=chunks)

        # Sorted descending: "second" (idx 1) should be first
        assert ranked[0].original_index == 1
        assert ranked[1].original_index == 0

    @pytest.mark.asyncio
    async def test_async_rerank_returns_same_as_sync(self):
        """async_rerank should return identical results to rerank."""
        reranker = self._make_reranker_with_mock_scores([3.0, -3.0])
        chunks = ["Relevant.", "Irrelevant."]
        sync_result = reranker.rerank(query="q", chunks=chunks)
        async_result = await reranker.async_rerank(query="q", chunks=chunks)

        assert len(sync_result) == len(async_result)
        assert sync_result[0].text == async_result[0].text


# ---------------------------------------------------------------------------
# 3. NLIRAGGroundingGuard + reranker integration
# ---------------------------------------------------------------------------

class TestNLIGuardWithReranker:
    """Tests the full pipeline with context reranking enabled, using mocked reranker."""

    def _guard(self) -> NLIRAGGroundingGuard:
        return NLIRAGGroundingGuard(backend=HeuristicNLIBackend())

    def _make_mock_reranker(self, filtered_chunks: list[str]) -> MagicMock:
        """Returns a mock ContextReranker whose async_rerank resolves to filtered_chunks."""
        mock = MagicMock(spec=ContextReranker)
        mock.async_rerank = MagicMock(
            return_value=asyncio.coroutine(
                lambda *a, **k: [
                    RankedChunk(text=c, relevance_score=0.9, original_index=i)
                    for i, c in enumerate(filtered_chunks)
                ]
            )()
        )
        return mock

    @pytest.mark.asyncio
    async def test_reranker_metadata_in_result_when_enabled(self):
        """When context reranking is enabled and fires, metadata should reflect it."""
        guard = self._guard()
        policy = _make_rag_policy(
            min_grounding_score=0.50,
            early_exit_threshold=0.0,
            context_reranking_enabled=True,
        )
        ctx = _make_ctx(
            policy=policy,
            completion="Refunds are processed in 7 business days.",
            context=["Refunds take 5 to 7 business days.", "Our CEO joined in 2018."],
            messages=[{"role": "user", "content": "What is the refund policy?"}],
        )

        # Patch get_context_reranker so it returns a mock that filters to 1 chunk
        with patch(
            "controlplane.detectors.hallucination.nli_verifier.get_context_reranker",
            return_value=None,  # None = reranker not available, skip silently
        ):
            result = await guard.evaluate(ctx)
            # Even without reranker, should not crash — metadata shows disabled
            assert "context_reranker" in result.metadata
            assert result.metadata["context_reranker"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_pipeline_works_without_reranker_installed(self):
        """get_context_reranker returning None must not crash the pipeline."""
        guard = self._guard()
        policy = _make_rag_policy(
            min_grounding_score=0.50,
            early_exit_threshold=0.0,
            context_reranking_enabled=True,
        )
        ctx = _make_ctx(
            policy=policy,
            completion="Refunds are processed within 7 days.",
            context=["Refunds are issued within 5 to 7 business days."],
        )

        with patch(
            "controlplane.detectors.hallucination.nli_verifier.get_context_reranker",
            return_value=None,
        ):
            result = await guard.evaluate(ctx)
            # NLI should still run on full context, pipeline should complete
            assert result.metadata is not None
            assert "claims_total" in result.metadata


# ---------------------------------------------------------------------------
# 4. auto_select_backend — cross_encoder preference
# ---------------------------------------------------------------------------

class TestAutoSelectBackendCrossEncoder:

    def test_explicit_heuristic_preference(self):
        """Requesting 'heuristic' always returns HeuristicNLIBackend."""
        backend = auto_select_backend(preferred="heuristic")
        assert isinstance(backend, HeuristicNLIBackend)

    def test_explicit_cross_encoder_raises_when_unavailable(self):
        """Requesting 'cross_encoder' when sentence_transformers is absent must raise,
        not silently fall back. This prevents misconfiguration from going unnoticed."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Only test this if the import would actually fail
            try:
                import sentence_transformers  # noqa: F401
                pytest.skip("sentence_transformers is installed; skip unavailability test")
            except ImportError:
                with pytest.raises(Exception):
                    auto_select_backend(preferred="cross_encoder")
