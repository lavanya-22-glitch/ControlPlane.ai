"""
Unit tests for:
  - CitationParser         (citation_parser.py)
  - NgramSimilarityScorer  (citation_similarity.py)
  - CitationGroundingGuard (citation_similarity.py)
    - disabled / no citations passthrough
    - per-citation short-circuit on fail_threshold
    - aggregate avg_threshold check
    - full audit metadata shape

All tests use NgramSimilarityScorer — no sentence-transformers required.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from controlplane.detectors.hallucination.citation_parser import Citation, CitationParser
from controlplane.detectors.hallucination.citation_similarity import (
    CitationGroundingGuard,
    CitationSimilarityResult,
    NgramSimilarityScorer,
)
from controlplane.detectors.base import GuardContext
from controlplane.policy.models import (
    HallucinationConfig,
    PolicyDefinition,
    PostExecutionConfig,
)
from controlplane.pdp.types import PDPAction, ViolationCode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rag_policy(
    enabled: bool = True,
    citation_check_enabled: bool = True,
    citation_fail_threshold: float = 0.35,
    citation_avg_threshold: float = 0.50,
    citation_similarity_method: str = "ngram",
    action: str = "rewrite",
) -> PolicyDefinition:
    return PolicyDefinition(
        mode="rag",
        post_execution=PostExecutionConfig(
            hallucination_engine=HallucinationConfig(
                enabled=enabled,
                min_grounding_score=0.70,
                action_on_low_grounding=action,
                citation_check_enabled=citation_check_enabled,
                citation_fail_threshold=citation_fail_threshold,
                citation_avg_threshold=citation_avg_threshold,
                citation_similarity_method=citation_similarity_method,
            )
        ),
    )


def _make_ctx(
    policy: PolicyDefinition,
    completion: str,
    context: list[str] | None,
    trace_id: str = "test_trace",
) -> GuardContext:
    return GuardContext(
        trace_id=trace_id,
        app_id="test",
        policy=policy,
        messages=[{"role": "user", "content": "What is the refund policy?"}],
        retrieved_context=context,
        completion_text=completion,
    )


# ---------------------------------------------------------------------------
# 1. CitationParser unit tests
# ---------------------------------------------------------------------------

class TestCitationParser:
    parser = CitationParser()

    def test_parses_numeric_citation(self):
        text = "Refunds take 7 business days. [1]"
        citations = self.parser.parse(text)
        assert len(citations) == 1
        assert citations[0].marker == "[1]"
        assert citations[0].marker_type == "numeric"
        assert "7 business days" in citations[0].preceding_text

    def test_parses_multi_numeric_citation(self):
        text = "Employees receive 30 days of annual leave. [1, 2]"
        citations = self.parser.parse(text)
        assert len(citations) == 1
        assert citations[0].marker_value == "1, 2"
        assert citations[0].is_multi_reference is True

    def test_parses_named_citation(self):
        text = "The policy states a 30-day return window. [HR Policy]"
        citations = self.parser.parse(text)
        assert any(c.marker_type == "named" for c in citations)
        named = [c for c in citations if c.marker_type == "named"][0]
        assert "HR Policy" in named.marker_value

    def test_parses_source_citation(self):
        text = "Insurance covers dental and vision. [Source: Benefits Guide 2024]"
        citations = self.parser.parse(text)
        assert len(citations) >= 1
        source_cit = [c for c in citations if c.marker_type == "source"]
        assert len(source_cit) == 1
        assert "Benefits Guide" in source_cit[0].marker_value

    def test_parses_author_year_citation(self):
        text = "The standard deviation is known to be stable. (Smith et al., 2023)"
        citations = self.parser.parse(text)
        assert any(c.marker_type == "author_year" for c in citations)

    def test_multiple_citations_in_order(self):
        text = (
            "Refunds take 5-7 days. [1] "
            "Employees are entitled to 30 days of leave. [2] "
            "Coverage includes dental. [3]"
        )
        citations = self.parser.parse(text)
        assert len(citations) == 3
        # Should be in document order
        positions = [c.position for c in citations]
        assert positions == sorted(positions)

    def test_empty_text_returns_empty(self):
        assert self.parser.parse("") == []

    def test_no_citations_returns_empty(self):
        assert self.parser.parse("There are no citation markers here at all.") == []

    def test_has_citations_quick_check(self):
        assert self.parser.has_citations("See details in [1].") is True
        assert self.parser.has_citations("No markers here.") is False

    def test_preceding_text_extracted_correctly(self):
        text = "This sentence should be the preceding text. [1] This should not be included."
        citations = self.parser.parse(text)
        assert len(citations) == 1
        # The preceding_text should be the sentence before [1], not after
        assert "preceding" in citations[0].preceding_text.lower()
        assert "should not" not in citations[0].preceding_text

    def test_citation_at_start_skipped(self):
        """A citation with no preceding text should not be returned."""
        citations = self.parser.parse("[1] This text comes AFTER the citation.")
        # No valid preceding text — citation should be skipped
        assert len(citations) == 0

    def test_marker_type_attribute(self):
        text = "Refunds take 7 days. [1]"
        citations = self.parser.parse(text)
        assert citations[0].marker_type == "numeric"


# ---------------------------------------------------------------------------
# 2. NgramSimilarityScorer unit tests
# ---------------------------------------------------------------------------

class TestNgramSimilarityScorer:
    scorer = NgramSimilarityScorer()

    def test_identical_texts_score_near_one(self):
        score = self.scorer.score("Refunds take 7 business days.", "Refunds take 7 business days.")
        assert score >= 0.90

    def test_similar_texts_score_high(self):
        score = self.scorer.score(
            "Refunds are processed within 7 business days.",
            "Refunds are issued within 5 to 7 business days.",
        )
        assert score >= 0.50, f"Expected ≥0.50, got {score}"

    def test_unrelated_texts_score_low(self):
        score = self.scorer.score(
            "Refunds are processed within 7 business days.",
            "The quarterly earnings report was released yesterday.",
        )
        assert score < 0.30, f"Expected <0.30, got {score}"

    def test_empty_inputs_return_zero(self):
        assert self.scorer.score("", "Some context") == 0.0
        assert self.scorer.score("Claimed text", "") == 0.0

    def test_scores_clamped_to_unit_interval(self):
        pairs = [
            ("some text here", "some text here"),
            ("a", "b"),
            ("", ""),
        ]
        for a, b in pairs:
            s = self.scorer.score(a, b)
            assert 0.0 <= s <= 1.0

    def test_score_all_chunks_returns_best(self):
        cited = "Refunds take 7 days."
        chunks = [
            "Quarterly earnings call is next week.",
            "Refunds are issued within 5 to 7 business days.",
            "The CEO joined the company in 2018.",
        ]
        best_chunk, best_idx, best_score = self.scorer.score_all_chunks(cited, chunks)
        assert best_idx == 1
        assert "Refunds" in best_chunk
        assert best_score > 0.40

    def test_score_all_chunks_empty_returns_defaults(self):
        chunk, idx, score = self.scorer.score_all_chunks("some text", [])
        assert chunk == ""
        assert idx == -1
        assert score == 0.0


# ---------------------------------------------------------------------------
# 3. CitationGroundingGuard integration tests
# ---------------------------------------------------------------------------

class TestCitationGroundingGuard:
    """Uses NgramSimilarityScorer injected directly — no models required."""

    def _guard(self) -> CitationGroundingGuard:
        return CitationGroundingGuard(scorer=NgramSimilarityScorer())

    @pytest.mark.asyncio
    async def test_guard_disabled_passes(self):
        """citation_check_enabled=False must always return PASS."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(citation_check_enabled=False),
            completion="Refunds take 7 days. [1]",
            context=["Refunds are issued within 5 to 7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is True
        assert result.metadata.get("reason") == "citation_check_disabled"

    @pytest.mark.asyncio
    async def test_no_citations_passes(self):
        """Completion with no citation markers should pass (no-op)."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(),
            completion="Refunds are typically processed within a week.",
            context=["Refunds take 5-7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is True
        assert result.metadata.get("reason") == "no_citations_found"

    @pytest.mark.asyncio
    async def test_grounded_citation_passes(self):
        """Citation whose preceding text closely matches context should PASS."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(citation_fail_threshold=0.20, citation_avg_threshold=0.20),
            completion="Refunds take 7 business days. [1]",
            context=["Refunds are issued within 5 to 7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_hallucinated_citation_short_circuits(self):
        """Citation whose preceding text is unrelated to context short-circuits immediately."""
        guard = self._guard()
        ctx = _make_ctx(
            # Very high threshold so even partial overlap fails
            policy=_make_rag_policy(citation_fail_threshold=0.95, citation_avg_threshold=0.95),
            completion="All users get complimentary 200% vouchers for premium flights. [1]",
            context=["Refunds are issued within 5 to 7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is False
        assert result.metadata.get("short_circuit") is True
        assert result.suggested_action == PDPAction.REWRITE

    @pytest.mark.asyncio
    async def test_block_action_when_configured(self):
        """action='block' should produce PDPAction.BLOCK on failure."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(
                citation_fail_threshold=0.99,
                action="block",
            ),
            completion="Completely unrelated content. [1]",
            context=["Refunds take 7 days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is False
        assert result.suggested_action == PDPAction.BLOCK

    @pytest.mark.asyncio
    async def test_violation_code_is_hallucination(self):
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(citation_fail_threshold=0.99),
            completion="Unrelated text about dragons. [1]",
            context=["Refunds are processed in 7 days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is False
        assert result.violation_code == ViolationCode.HALLUCINATION

    @pytest.mark.asyncio
    async def test_multiple_citations_short_circuit_on_first_failure(self):
        """With a high threshold, the first citation should short-circuit.
        The metadata should show only 1 citation evaluated, not all."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(citation_fail_threshold=0.99, citation_avg_threshold=0.99),
            completion=(
                "Completely unrelated statement one. [1] "
                "Completely unrelated statement two. [2] "
                "Completely unrelated statement three. [3]"
            ),
            context=["Refunds are processed in 7 days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is False
        assert result.metadata.get("short_circuit") is True
        # Should have stopped after the first failing citation
        assert result.metadata.get("citations_evaluated", 99) <= 1

    @pytest.mark.asyncio
    async def test_avg_threshold_check_after_all_pass_individually(self):
        """If no single citation short-circuits but avg is below avg_threshold, fail."""
        guard = self._guard()
        ctx = _make_ctx(
            # Low per-citation threshold (no short-circuit) but high avg threshold
            policy=_make_rag_policy(
                citation_fail_threshold=0.0,
                citation_avg_threshold=0.99,
            ),
            completion=(
                "Some claim about something unrelated. [1] "
                "Another unrelated claim. [2]"
            ),
            context=["Refunds are processed in 7 days."],
        )
        result = await guard.evaluate(ctx)
        # Both citations individually pass the 0.0 threshold, but avg < 0.99
        assert result.passed is False
        assert result.metadata.get("short_circuit") is False

    @pytest.mark.asyncio
    async def test_audit_metadata_shape(self):
        """Full metadata structure must always be present for audit traceability."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(
                citation_fail_threshold=0.0,
                citation_avg_threshold=0.0,
            ),
            completion="Refunds take 7 days. [1]",
            context=["Refunds are issued within 5 to 7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is True
        meta = result.metadata
        assert "citations_total" in meta
        assert "citations_evaluated" in meta
        assert "avg_similarity_score" in meta
        assert "all_citation_scores" in meta
        assert isinstance(meta["all_citation_scores"], list)
        first = meta["all_citation_scores"][0]
        assert "marker" in first
        assert "similarity_score" in first
        assert "is_grounded" in first
        assert "preceding_text" in first

    @pytest.mark.asyncio
    async def test_no_context_passes(self):
        """No retrieved_context → guard must pass (nothing to verify against)."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(),
            completion="Some response with a citation. [1]",
            context=None,
        )
        result = await guard.evaluate(ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_hallucination_engine_disabled_passes(self):
        """If top-level enabled=False, citation check must also pass."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(enabled=False, citation_check_enabled=True),
            completion="Unrelated content. [1]",
            context=["Refunds take 7 days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is True


# ---------------------------------------------------------------------------
# 4. NgramSimilarityScorer edge cases
# ---------------------------------------------------------------------------

class TestNgramEdgeCases:
    scorer = NgramSimilarityScorer()

    def test_stopword_heavy_claim_does_not_artificially_inflate(self):
        """Stopword filtering should prevent 'the a is are' from matching anything."""
        score = self.scorer.score(
            "the a is are was were be been",
            "Refunds are processed within seven business days",
        )
        # Should be low — no content word overlap after stopword removal
        assert score < 0.40

    def test_numbers_treated_as_content_words(self):
        """Numeric tokens should contribute to content-word overlap."""
        score_match = self.scorer.score(
            "Refunds take 7 days.",
            "Refunds are issued within 7 business days.",
        )
        score_mismatch = self.scorer.score(
            "Refunds take 7 days.",
            "Refunds are issued within 30 business days.",
        )
        assert score_match > score_mismatch
