"""
Unit tests for ChatbotHallucinationGuard.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
import json

from controlplane.detectors.hallucination.chatbot_judge import ChatbotHallucinationGuard
from controlplane.detectors.base import GuardContext
from controlplane.policy.models import PolicyDefinition, PostExecutionConfig, ChatbotHallucinationConfig
from controlplane.pdp.types import PDPAction, ViolationCode


def _make_policy(enabled: bool = True, action: str = "rewrite") -> PolicyDefinition:
    return PolicyDefinition(
        mode="chatbot",
        post_execution=PostExecutionConfig(
            chatbot_hallucination=ChatbotHallucinationConfig(
                enabled=enabled,
                action_on_violation=action
            )
        )
    )

def _make_ctx(policy: PolicyDefinition, completion: str = "Test response", messages: list = None) -> GuardContext:
    if messages is None:
        messages = [{"role": "user", "content": "Hello"}]
    return GuardContext(
        trace_id="test_trace",
        app_id="test",
        policy=policy,
        messages=messages,
        completion_text=completion,
    )

class TestChatbotHallucinationGuard:

    @pytest.mark.asyncio
    async def test_guard_disabled(self):
        guard = ChatbotHallucinationGuard()
        ctx = _make_ctx(_make_policy(enabled=False))
        res = await guard.evaluate(ctx)
        assert res.passed is True
        assert res.metadata["reason"] == "chatbot_hallucination_check_disabled"

    @pytest.mark.asyncio
    async def test_no_completion_text(self):
        guard = ChatbotHallucinationGuard()
        ctx = _make_ctx(_make_policy(), completion="")
        res = await guard.evaluate(ctx)
        assert res.passed is True
        assert res.metadata["reason"] == "no_completion_text"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_api_call_success_no_hallucination(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": '{"is_hallucination": false, "reason": "Consistent with history"}'
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        guard = ChatbotHallucinationGuard()
        ctx = _make_ctx(_make_policy())
        res = await guard.evaluate(ctx)
        
        assert res.passed is True
        assert res.metadata["is_hallucination"] is False

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_api_call_success_hallucination_detected(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": '{"is_hallucination": true, "reason": "Contradicts previous user statement"}'
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        guard = ChatbotHallucinationGuard()
        ctx = _make_ctx(_make_policy(action="block"))
        res = await guard.evaluate(ctx)
        
        assert res.passed is False
        assert res.violation_code == ViolationCode.HALLUCINATION
        assert res.suggested_action == PDPAction.BLOCK
        assert res.metadata["is_hallucination"] is True
        assert "Contradicts" in res.reason

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_api_call_invalid_json(self, mock_post):
        mock_resp = MagicMock()
        # Mocking an SLM that failed to output valid JSON
        mock_resp.json.return_value = {
            "response": "Yes it is hallucinating"
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        guard = ChatbotHallucinationGuard()
        ctx = _make_ctx(_make_policy())
        res = await guard.evaluate(ctx)
        
        # Should fail-open
        assert res.passed is True
        assert res.metadata["reason"] == "judge_output_parse_error"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_api_call_network_error(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        guard = ChatbotHallucinationGuard()
        ctx = _make_ctx(_make_policy())
        res = await guard.evaluate(ctx)
        
        # Should fail-open
        assert res.passed is True
        assert res.metadata["reason"] == "ollama_api_error"
