"""
Unit tests for ConfidenceMismatchGuard.
"""
import pytest
from typing import Dict, Any

from controlplane.detectors.tool_safety.confidence_mismatch import ConfidenceMismatchGuard
from controlplane.detectors.base import GuardContext
from controlplane.policy.models import PolicyDefinition, AgentConfig, ConfidenceMismatchConfig
from controlplane.pdp.types import PDPAction, ViolationCode


def _make_policy(enabled: bool = True, avg_thresh: float = -2.0, min_thresh: float = -4.0, per_token_thresh: float = -2.5, max_low_conf: float = 0.3) -> PolicyDefinition:
    return PolicyDefinition(
        mode="agent",
        agent_config=AgentConfig(
            confidence_mismatch=ConfidenceMismatchConfig(
                enabled=enabled,
                avg_logprob_threshold=avg_thresh,
                min_logprob_threshold=min_thresh,
                per_token_threshold=per_token_thresh,
                max_low_conf_pct=max_low_conf,
            )
        )
    )

def _make_ctx(policy: PolicyDefinition, raw_response: Dict[str, Any], tool_calls: list = None) -> GuardContext:
    ctx = GuardContext(
        trace_id="test_trace",
        app_id="test",
        policy=policy,
        tool_calls=tool_calls,
    )
    ctx.metadata["raw_response"] = raw_response
    return ctx

class TestConfidenceMismatchGuard:
    
    @pytest.mark.asyncio
    async def test_guard_disabled(self):
        guard = ConfidenceMismatchGuard()
        ctx = _make_ctx(_make_policy(enabled=False), {})
        res = await guard.evaluate(ctx)
        assert res.passed is True
        assert res.metadata["reason"] == "confidence_check_disabled"

    @pytest.mark.asyncio
    async def test_no_tool_calls(self):
        guard = ConfidenceMismatchGuard()
        ctx = _make_ctx(_make_policy(), {}, tool_calls=None)
        res = await guard.evaluate(ctx)
        assert res.passed is True
        assert res.metadata["reason"] == "no_tool_calls"

    @pytest.mark.asyncio
    async def test_no_logprobs(self):
        guard = ConfidenceMismatchGuard()
        ctx = _make_ctx(_make_policy(), {}, tool_calls=[{"function": {"name": "test"}}])
        res = await guard.evaluate(ctx)
        assert res.passed is True
        assert res.metadata["reason"] == "logprobs_unavailable"

    @pytest.mark.asyncio
    async def test_high_confidence_passes(self):
        guard = ConfidenceMismatchGuard()
        raw_response = {
            "choices": [{
                "logprobs": {
                    "content": [
                        {"token": "arg1", "logprob": -0.1},
                        {"token": "arg2", "logprob": -0.2},
                    ]
                }
            }]
        }
        ctx = _make_ctx(_make_policy(), raw_response, tool_calls=[{"function": {"name": "test"}}])
        res = await guard.evaluate(ctx)
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_low_avg_confidence_fails(self):
        guard = ConfidenceMismatchGuard()
        raw_response = {
            "choices": [{
                "logprobs": {
                    "content": [
                        {"token": "arg1", "logprob": -3.0},
                        {"token": "arg2", "logprob": -3.5},
                    ]
                }
            }]
        }
        ctx = _make_ctx(_make_policy(avg_thresh=-2.0), raw_response, tool_calls=[{"function": {"name": "test"}}])
        res = await guard.evaluate(ctx)
        assert res.passed is False
        assert res.violation_code == ViolationCode.AGENT_CONFIDENCE_MISMATCH
        assert res.suggested_action == PDPAction.BLOCK
        assert "avg_logprob" in res.reason

    @pytest.mark.asyncio
    async def test_min_logprob_fails(self):
        guard = ConfidenceMismatchGuard()
        raw_response = {
            "choices": [{
                "logprobs": {
                    "content": [
                        {"token": "arg1", "logprob": -0.1},
                        {"token": "arg2", "logprob": -5.0}, # below min_thresh -4.0
                    ]
                }
            }]
        }
        # Average is -2.55, min is -5.0. 
        # pct_low is 0.5 (1/2), if we set max_low to 1.0 to avoid that trigger
        ctx = _make_ctx(_make_policy(avg_thresh=-3.0, min_thresh=-4.0, max_low_conf=1.0), raw_response, tool_calls=[{"function": {"name": "test"}}])
        res = await guard.evaluate(ctx)
        assert res.passed is False
        assert "min_logprob" in res.reason

    @pytest.mark.asyncio
    async def test_pct_low_conf_fails(self):
        guard = ConfidenceMismatchGuard()
        raw_response = {
            "choices": [{
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": -0.1},
                        {"token": "b", "logprob": -0.1},
                        {"token": "c", "logprob": -2.6}, # below per_token -2.5
                        {"token": "d", "logprob": -2.6}, # below per_token -2.5
                    ]
                }
            }]
        }
        # pct_low is 0.5. max_low_conf is 0.3. Avg is (-5.4 / 4) = -1.35
        ctx = _make_ctx(_make_policy(avg_thresh=-2.0, min_thresh=-4.0, max_low_conf=0.3, per_token_thresh=-2.5), raw_response, tool_calls=[{"function": {"name": "test"}}])
        res = await guard.evaluate(ctx)
        assert res.passed is False
        assert "tokens below per_token_threshold" in res.reason
