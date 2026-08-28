"""
Unit tests for ReasoningOutputSimilarityGuard.
"""
import pytest
from typing import Dict, Any

from controlplane.detectors.tool_safety.reasoning_similarity import ReasoningOutputSimilarityGuard, NgramReasoningScorer
from controlplane.detectors.base import GuardContext
from controlplane.policy.models import PolicyDefinition, AgentConfig, ReasoningSimilarityConfig
from controlplane.pdp.types import PDPAction, ViolationCode


def _make_policy(enabled: bool = True, method: str = "ngram", fail_thresh: float = 0.15) -> PolicyDefinition:
    return PolicyDefinition(
        mode="agent",
        agent_config=AgentConfig(
            reasoning_similarity=ReasoningSimilarityConfig(
                enabled=enabled,
                similarity_method=method,
                similarity_fail_threshold=fail_thresh,
            )
        )
    )

def _make_ctx(policy: PolicyDefinition, completion: str = None, metadata: Dict = None, tool_calls: list = None) -> GuardContext:
    ctx = GuardContext(
        trace_id="test_trace",
        app_id="test",
        policy=policy,
        completion_text=completion,
        tool_calls=tool_calls,
    )
    if metadata:
        ctx.metadata.update(metadata)
    return ctx


class TestReasoningOutputSimilarityGuard:

    @pytest.mark.asyncio
    async def test_guard_disabled(self):
        guard = ReasoningOutputSimilarityGuard()
        ctx = _make_ctx(_make_policy(enabled=False))
        res = await guard.evaluate(ctx)
        assert res.passed is True
        assert res.metadata["reason"] == "reasoning_similarity_check_disabled"

    @pytest.mark.asyncio
    async def test_no_tool_calls(self):
        guard = ReasoningOutputSimilarityGuard()
        ctx = _make_ctx(_make_policy(), tool_calls=[])
        res = await guard.evaluate(ctx)
        assert res.passed is True
        assert res.metadata["reason"] == "no_tool_calls"

    @pytest.mark.asyncio
    async def test_no_reasoning_text(self):
        guard = ReasoningOutputSimilarityGuard()
        ctx = _make_ctx(_make_policy(), completion="Just calling a tool", tool_calls=[{"function": {"name": "test"}}])
        res = await guard.evaluate(ctx)
        assert res.passed is True
        assert res.metadata["reason"] == "reasoning_unavailable"

    @pytest.mark.asyncio
    async def test_aligned_reasoning_passes(self):
        guard = ReasoningOutputSimilarityGuard(scorer=NgramReasoningScorer())
        # The reasoning and the args have similar content words (execute_command, read_file, file_path)
        ctx = _make_ctx(
            _make_policy(fail_thresh=0.1),
            metadata={"reasoning_text": "I will execute_command to read_file on file_path config.json."},
            tool_calls=[{"function": {"name": "execute_command", "arguments": '{"command": "read_file", "file_path": "config.json"}'}}]
        )
        res = await guard.evaluate(ctx)
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_misaligned_reasoning_fails(self):
        guard = ReasoningOutputSimilarityGuard(scorer=NgramReasoningScorer())
        ctx = _make_ctx(
            _make_policy(fail_thresh=0.15),
            metadata={"reasoning_text": "I should probably query the database for users."},
            tool_calls=[{"function": {"name": "execute_command", "arguments": '{"command": "rm -rf /"}'}}]
        )
        res = await guard.evaluate(ctx)
        assert res.passed is False
        assert res.violation_code == ViolationCode.AGENT_REASONING_MISMATCH
        assert res.suggested_action == PDPAction.BLOCK

    @pytest.mark.asyncio
    async def test_extracts_reasoning_from_tags(self):
        guard = ReasoningOutputSimilarityGuard(scorer=NgramReasoningScorer())
        completion = "<thinking>I will delete the user.</thinking>\n..."
        ctx = _make_ctx(
            _make_policy(fail_thresh=0.15),
            completion=completion,
            tool_calls=[{"function": {"name": "delete_user", "arguments": '{"user_id": 123}'}}]
        )
        res = await guard.evaluate(ctx)
        assert res.passed is True
        assert "delete the user" in res.metadata["reasoning_preview"].lower()

    @pytest.mark.asyncio
    async def test_extracts_reasoning_from_raw_response(self):
        guard = ReasoningOutputSimilarityGuard(scorer=NgramReasoningScorer())
        raw_resp = {
            "choices": [{
                "message": {
                    "reasoning_content": "I need to update the order status."
                }
            }]
        }
        ctx = _make_ctx(
            _make_policy(fail_thresh=0.1),
            metadata={"raw_response": raw_resp},
            tool_calls=[{"function": {"name": "update_order", "arguments": '{"status": "shipped"}'}}]
        )
        res = await guard.evaluate(ctx)
        assert res.passed is True
        assert "update the order status" in res.metadata["reasoning_preview"].lower()
