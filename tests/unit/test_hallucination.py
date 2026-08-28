"""
Unit tests for the NLI RAG Grounding Engine.

Coverage:
    - HeuristicNLIBackend.score_pairs()              — grounded / contradicted / neutral
    - NLIRAGGroundingGuard.evaluate():
        - grounded response (should PASS)
        - hallucinated response (should FAIL → REWRITE)
        - lexical pre-filter gate (high-overlap claims skip NLI)
        - per-claim early exit short-circuit
        - no context supplied (should PASS, no-op)
        - hallucination engine disabled (should PASS, no-op)
        - auto_select_backend() returns HeuristicNLIBackend in test env
"""

import pytest

from controlplane.detectors.hallucination.nli_backend import (
    HeuristicNLIBackend,
    NLIResult,
    auto_select_backend,
)
from controlplane.detectors.hallucination.nli_verifier import NLIRAGGroundingGuard
from controlplane.detectors.hallucination.context_aligner import align_claims_to_context
from controlplane.detectors.hallucination.claim_splitter import split_claims
from controlplane.detectors.base import GuardContext
from controlplane.policy.models import HallucinationConfig, PolicyDefinition, PostExecutionConfig
from controlplane.pdp.types import PDPAction, ViolationCode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rag_policy(
    min_grounding_score: float = 0.70,
    early_exit_threshold: float = 0.20,
    action: str = "rewrite",
    enabled: bool = True,
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
        retrieved_context=context,
        completion_text=completion,
    )


# ---------------------------------------------------------------------------
# 1. HeuristicNLIBackend unit tests
# ---------------------------------------------------------------------------

class TestHeuristicNLIBackend:
    backend = HeuristicNLIBackend()

    def test_grounded_claim_high_overlap(self):
        """Claim that closely mirrors context should produce high entailment."""
        results = self.backend.score_pairs([
            ("Refunds are issued within 5 to 7 business days.", 
             "Refunds are processed within 5 to 7 business days."),
        ])
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, NLIResult)
        assert r.grounding_score >= 0.70, f"Expected grounding ≥ 0.70, got {r.grounding_score}"

    def test_contradicted_claim_negation_mismatch(self):
        """Claim that negates context should score lower than grounded claim."""
        grounded = self.backend.score_pairs([
            ("The service is available 24/7.", "The service is available 24/7."),
        ])[0]
        contradicted = self.backend.score_pairs([
            ("The service is available 24/7.", "The service is never available on weekends."),
        ])[0]
        assert contradicted.grounding_score < grounded.grounding_score

    def test_unrelated_claim_low_score(self):
        """Completely unrelated claim should score much lower than grounded."""
        results = self.backend.score_pairs([
            ("Refunds are issued within 5 to 7 business days.",
             "All customers receive 200% airline vouchers valid for 100 years."),
        ])
        assert results[0].grounding_score < 0.30

    def test_empty_pairs_returns_empty(self):
        assert self.backend.score_pairs([]) == []

    def test_batch_preserves_order(self):
        """Results must be returned in the same order as input pairs."""
        pairs = [
            ("Context A with apples and oranges.", "This is about apples and oranges."),
            ("Refunds take 7 days.", "Customers get free flights."),
        ]
        results = self.backend.score_pairs(pairs)
        assert len(results) == 2
        # First pair should score higher than second
        assert results[0].grounding_score > results[1].grounding_score

    def test_grounding_score_clamped_to_unit_interval(self):
        """grounding_score must always be in [0, 1]."""
        pairs = [
            ("x", "x x x x x x x"),
            ("", ""),
            ("something", "completely different"),
        ]
        for r in self.backend.score_pairs(pairs):
            assert 0.0 <= r.grounding_score <= 1.0


# ---------------------------------------------------------------------------
# 2. NLIRAGGroundingGuard integration tests
# ---------------------------------------------------------------------------

class TestNLIRAGGroundingGuard:
    """All tests inject HeuristicNLIBackend to avoid requiring Transformers/ONNX."""

    def _guard(self) -> NLIRAGGroundingGuard:
        return NLIRAGGroundingGuard(backend=HeuristicNLIBackend())

    @pytest.mark.asyncio
    async def test_grounded_response_passes(self):
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(min_grounding_score=0.50),
            completion="Refunds are typically processed within 5 to 7 business days.",
            context=["Refunds are issued within 5 to 7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is True
        assert result.suggested_action == PDPAction.ALLOW

    @pytest.mark.asyncio
    async def test_hallucinated_response_fails(self):
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(min_grounding_score=0.70),
            completion="All customers receive 200% airline vouchers valid for 100 years immediately.",
            context=["Refunds are issued within 5 to 7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is False
        assert result.suggested_action == PDPAction.REWRITE
        assert result.violation_code == ViolationCode.HALLUCINATION
        assert result.score is not None and result.score < 0.70

    @pytest.mark.asyncio
    async def test_block_action_when_configured(self):
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(min_grounding_score=0.70, action="block"),
            completion="This has absolutely nothing to do with the context at all ever.",
            context=["Refunds are issued within 5 to 7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is False
        assert result.suggested_action == PDPAction.BLOCK

    @pytest.mark.asyncio
    async def test_no_context_allows(self):
        """When no retrieved context is provided, guard should pass (nothing to verify against)."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(),
            completion="Some completion text.",
            context=None,
        )
        result = await guard.evaluate(ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_completion_allows(self):
        """Empty completion — nothing to evaluate."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(),
            completion="",
            context=["Some context."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_engine_disabled_allows(self):
        """Disabled hallucination engine should short-circuit immediately."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(enabled=False),
            completion="Completely made up nonsense about dragons and magic carpets.",
            context=["Refunds are issued within 5 to 7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_early_exit_fires_on_first_bad_claim(self):
        """A claim with grounding score below early_exit_threshold triggers early exit."""
        guard = self._guard()
        # Set a very HIGH early_exit_threshold so even mediocre claims trigger it
        ctx = _make_ctx(
            policy=_make_rag_policy(min_grounding_score=0.70, early_exit_threshold=0.99),
            completion="Completely unrelated claim about dragons. Another unrelated fact.",
            context=["Refunds are issued within 5 to 7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert result.passed is False
        assert result.metadata.get("early_exit") is True

    @pytest.mark.asyncio
    async def test_metadata_contains_per_claim_scores(self):
        """Audit metadata must always include per_claim_scores for traceability."""
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(min_grounding_score=0.50, early_exit_threshold=0.0),
            completion="Refunds take up to 7 business days.",
            context=["Refunds are issued within 5 to 7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert "per_claim_scores" in result.metadata
        claims_meta = result.metadata["per_claim_scores"]
        assert isinstance(claims_meta, list) and len(claims_meta) > 0
        first = claims_meta[0]
        assert "claim" in first
        assert "grounding_score" in first
        assert "entailment" in first
        assert "contradiction" in first

    @pytest.mark.asyncio
    async def test_metadata_contains_backend_name(self):
        guard = self._guard()
        ctx = _make_ctx(
            policy=_make_rag_policy(early_exit_threshold=0.0),
            completion="Refunds take 7 days.",
            context=["Refunds are issued within 5 to 7 business days."],
        )
        result = await guard.evaluate(ctx)
        assert result.metadata.get("backend") == "heuristic"


# ---------------------------------------------------------------------------
# 3. auto_select_backend() tests
# ---------------------------------------------------------------------------

class TestAutoSelectBackend:
    def test_auto_returns_heuristic_when_transformers_unavailable(self):
        """In the test environment, Transformers likely IS available but we
        can explicitly request the heuristic backend."""
        backend = auto_select_backend(preferred="heuristic")
        assert isinstance(backend, HeuristicNLIBackend)
        assert backend.backend_name == "heuristic"

    def test_heuristic_backend_warms_up_without_error(self):
        backend = HeuristicNLIBackend()
        backend.warmup()  # should be a no-op, must not raise


# ---------------------------------------------------------------------------
# 4. Context aligner unit tests
# ---------------------------------------------------------------------------

class TestContextAligner:
    def test_returns_one_pair_per_claim(self):
        claims = ["Refunds take 7 days.", "No hidden fees apply."]
        context = ["Refunds are issued within 7 business days.", "All fees are disclosed upfront."]
        pairs = align_claims_to_context(claims, context)
        assert len(pairs) == len(claims)

    def test_empty_context_returns_empty_overlap(self):
        pairs = align_claims_to_context(["Some claim."], [])
        assert len(pairs) == 1
        assert pairs[0].lexical_overlap == 0.0
        assert pairs[0].context_chunk == ""

    def test_selects_best_matching_chunk(self):
        claims = ["The refund takes 7 days."]
        context = [
            "Our return policy is 30 days.",
            "Refunds are processed within 7 business days.",
            "Contact support for help.",
        ]
        pairs = align_claims_to_context(claims, context)
        # The second chunk has the highest overlap (refund, 7, days)
        assert "7" in pairs[0].context_chunk or "refund" in pairs[0].context_chunk.lower()

    def test_overlap_values_are_in_unit_interval(self):
        pairs = align_claims_to_context(
            ["Any claim text here."],
            ["Some matching context with any overlapping words."],
        )
        assert 0.0 <= pairs[0].lexical_overlap <= 1.0


# ---------------------------------------------------------------------------
# 5. Claim splitter unit tests
# ---------------------------------------------------------------------------

class TestClaimSplitter:
    def test_splits_on_sentence_boundaries(self):
        text = "Refunds take 7 days. Support is 24/7. No hidden fees."
        claims = split_claims(text)
        assert len(claims) >= 2

    def test_filters_very_short_phrases(self):
        claims = split_claims("OK. Fine. Refunds are issued within 7 business days.")
        assert all(len(c.split()) >= 3 for c in claims)

    def test_empty_input(self):
        assert split_claims("") == []

    def test_strips_markdown(self):
        text = "## Summary\n- Refunds take 7 days.\n* Support is available 24/7."
        claims = split_claims(text)
        assert not any(c.startswith("#") or c.startswith("-") or c.startswith("*") for c in claims)
