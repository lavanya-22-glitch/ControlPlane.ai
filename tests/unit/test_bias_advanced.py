"""
Unit tests for advanced BiasSubspaceGuard features (Projections & Counterfactuals).
"""
import pytest
from unittest.mock import patch, MagicMock

from controlplane.detectors.bias_toxicity.bias_subspace import BiasSubspaceGuard
from controlplane.detectors.base import GuardContext
from controlplane.policy.models import PolicyDefinition, PostExecutionConfig
from controlplane.pdp.types import PDPAction, ViolationCode


def _make_policy(proj_enabled: bool, cf_enabled: bool) -> PolicyDefinition:
    return PolicyDefinition(
        mode="chatbot",
        post_execution=PostExecutionConfig(
            bias_subspace_threshold=0.65,
            bias_projection_enabled=proj_enabled,
            counterfactual_check_enabled=cf_enabled,
            counterfactual_ollama_base_url="http://mocked",
            action_on_violation="rewrite"
        )
    )

def _make_ctx(policy: PolicyDefinition, text: str) -> GuardContext:
    return GuardContext(
        trace_id="test",
        app_id="test",
        policy=policy,
        completion_text=text,
    )

class TestAdvancedBiasGuard:

    @pytest.mark.asyncio
    @patch("controlplane.detectors.bias_toxicity.bias_subspace.BiasProjectionScorer.score")
    async def test_bias_projection_flags_violation(self, mock_score):
        # Mock projection returning high skew on Gender axis
        mock_score.return_value = (0.80, "Latent Gender bias projection detected.")
        
        guard = BiasSubspaceGuard()
        ctx = _make_ctx(_make_policy(proj_enabled=True, cf_enabled=False), "The nurse was very helpful.")
        
        res = await guard.evaluate(ctx)
        
        assert res.passed is False
        assert res.violation_code == ViolationCode.BIAS
        assert res.suggested_action == PDPAction.REWRITE
        assert "Latent Gender bias projection" in res.reason

    @pytest.mark.asyncio
    @patch("controlplane.detectors.bias_toxicity.counterfactual.CounterfactualScorer.score")
    async def test_counterfactual_flags_violation(self, mock_cf_score):
        # Mock counterfactual finding bias after identity swap
        mock_cf_score.return_value = (True, "Counterfactual bias detected: Text becomes offensive when gender swapped.")
        
        guard = BiasSubspaceGuard()
        ctx = _make_ctx(_make_policy(proj_enabled=False, cf_enabled=True), "He is a terrible driver.")
        
        res = await guard.evaluate(ctx)
        
        assert res.passed is False
        assert res.violation_code == ViolationCode.BIAS
        assert res.metadata.get("counterfactual") is True
        assert "Text becomes offensive" in res.reason

    @pytest.mark.asyncio
    @patch("controlplane.detectors.bias_toxicity.counterfactual.CounterfactualScorer.score")
    async def test_counterfactual_passes(self, mock_cf_score):
        mock_cf_score.return_value = (False, "")
        
        guard = BiasSubspaceGuard()
        ctx = _make_ctx(_make_policy(proj_enabled=False, cf_enabled=True), "He is a great software engineer.")
        
        res = await guard.evaluate(ctx)
        
        assert res.passed is True
        
    @pytest.mark.asyncio
    async def test_both_disabled_passes(self):
        guard = BiasSubspaceGuard()
        ctx = _make_ctx(_make_policy(proj_enabled=False, cf_enabled=False), "Just a normal sentence.")
        
        res = await guard.evaluate(ctx)
        
        assert res.passed is True
