"""
Unit tests for Token Optimization guards:
- ContextCapperGuard (RAG context truncation)
- ToolDeduplicationGuard (Agent duplicate call short-circuit)
"""
import pytest
import json

from controlplane.detectors.token_optim.context_capper import ContextCapperGuard
from controlplane.detectors.token_optim.tool_deduplicator import ToolDeduplicationGuard
from controlplane.detectors.base import GuardContext
from controlplane.policy.models import PolicyDefinition, PreExecutionConfig, ContextCappingConfig, AgentConfig, ToolDeduplicationConfig
from controlplane.pdp.types import PDPAction, ViolationCode


def _make_rag_ctx(context_list: list, max_len: int, enabled: bool) -> GuardContext:
    policy = PolicyDefinition(
        mode="rag",
        pre_execution=PreExecutionConfig(
            context_capping=ContextCappingConfig(enabled=enabled, max_context_length_chars=max_len)
        )
    )
    return GuardContext(
        trace_id="test",
        app_id="test",
        policy=policy,
        retrieved_context=context_list
    )

class TestContextCapperGuard:
    
    @pytest.mark.asyncio
    async def test_truncates_long_context(self):
        guard = ContextCapperGuard()
        ctx = _make_rag_ctx(["A" * 2500, "B" * 2500], 4000, True)
        
        res = await guard.evaluate(ctx)
        
        assert res.passed is True
        assert res.suggested_action == PDPAction.TRANSFORM
        total_len = sum(len(c) for c in ctx.retrieved_context)
        assert total_len > 4000
        assert "Context truncated due to token optimization limits" in ctx.retrieved_context[-1]

    @pytest.mark.asyncio
    async def test_allows_short_context(self):
        guard = ContextCapperGuard()
        ctx = _make_rag_ctx(["Short text"], 4000, True)
        
        res = await guard.evaluate(ctx)
        
        assert res.passed is True
        assert res.suggested_action == PDPAction.ALLOW
        assert ctx.retrieved_context == ["Short text"]


def _make_agent_ctx(messages: list, tool_calls: list, max_dupes: int, enabled: bool) -> GuardContext:
    policy = PolicyDefinition(
        mode="agent",
        agent_config=AgentConfig(
            tool_deduplication=ToolDeduplicationConfig(enabled=enabled, max_duplicate_calls=max_dupes)
        )
    )
    return GuardContext(
        trace_id="test",
        app_id="test",
        policy=policy,
        messages=messages,
        tool_calls=tool_calls,
        completion_text="Thinking about calling the tool again..."
    )

class TestToolDeduplicationGuard:
    
    @pytest.mark.asyncio
    async def test_detects_and_blocks_duplicate(self):
        guard = ToolDeduplicationGuard()
        
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": '{"loc": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny and 75F"}
        ]
        
        # LLM tries to call get_weather("Paris") again
        current_tool_calls = [{"id": "call_2", "function": {"name": "get_weather", "arguments": '{"loc": "Paris"}'}}]
        
        ctx = _make_agent_ctx(messages, current_tool_calls, max_dupes=1, enabled=True)
        
        res = await guard.evaluate(ctx)
        
        assert res.passed is False
        assert res.violation_code == ViolationCode.DUPLICATE_TOOL_CALL
        assert res.suggested_action == PDPAction.REWRITE
        
        fallback = res.fallback_payload
        assert fallback is not None
        content = fallback["choices"][0]["message"]["content"]
        assert "Sunny and 75F" in content
        assert "Thinking about calling the tool again..." in content

    @pytest.mark.asyncio
    async def test_allows_under_threshold(self):
        guard = ToolDeduplicationGuard()
        
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": '{"loc": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny and 75F"}
        ]
        
        current_tool_calls = [{"id": "call_2", "function": {"name": "get_weather", "arguments": '{"loc": "Paris"}'}}]
        
        # Threshold is 2, this is only the 2nd attempt, so it should allow
        ctx = _make_agent_ctx(messages, current_tool_calls, max_dupes=2, enabled=True)
        
        res = await guard.evaluate(ctx)
        
        assert res.passed is True
        assert res.suggested_action == PDPAction.ALLOW
