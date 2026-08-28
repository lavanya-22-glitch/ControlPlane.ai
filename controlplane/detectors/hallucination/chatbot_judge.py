"""
SLM as a Judge for Chatbot Hallucination.

Detects hallucinations or fabrications in standalone chatbot responses
by consulting an external Small Language Model (SLM) via the Ollama API.
Since chatbots typically do not use retrieved context chunks, this guard
evaluates the model's completion against the conversation history for
internal consistency, fabrications, or logical contradictions.
"""
from __future__ import annotations

import json
import logging
import httpx
from typing import Dict, Any, Optional

from controlplane.detectors.base import BaseDetector, DetectionResult, GuardContext, GuardStage
from controlplane.pdp.types import PDPAction, ViolationCode

logger = logging.getLogger("controlplane.detectors.hallucination.chatbot_judge")

# The prompt used to instruct the SLM judge. We ask for a strict JSON output.
_JUDGE_PROMPT_TEMPLATE = """
You are an expert fact-checker and safety judge. 
Your task is to evaluate the final Assistant Response for hallucinations, fabrications, or severe logical contradictions based solely on the provided Conversation History.

Conversation History:
{history}

Assistant Response:
{response}

Is the Assistant Response hallucinating, fabricating facts, or contradicting the conversation history?
Output a strict JSON object with exactly two keys:
"is_hallucination": boolean (true if it hallucinates or fabricates, false otherwise)
"reason": string (a brief explanation of your decision)

JSON Output:
"""


class ChatbotHallucinationGuard(BaseDetector):
    """
    Evaluates chatbot responses using a local Ollama model (SLM as a judge).
    """
    
    # Use a single shared httpx client for connection pooling
    _client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if ChatbotHallucinationGuard._client is None:
            ChatbotHallucinationGuard._client = httpx.AsyncClient(timeout=10.0)
        return ChatbotHallucinationGuard._client

    @property
    def name(self) -> str:
        return "chatbot_hallucination_guard"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.POST_EXECUTION

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        if not ctx.policy.post_execution:
            return self._allow(metadata={"reason": "post_execution_disabled"})

        config = getattr(ctx.policy.post_execution, "chatbot_hallucination", None)
        if config is None or not getattr(config, "enabled", False):
            return self._allow(metadata={"reason": "chatbot_hallucination_check_disabled"})

        if not ctx.completion_text:
            return self._allow(metadata={"reason": "no_completion_text"})

        # Format the conversation history
        history_lines = []
        for msg in ctx.messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            history=history_text,
            response=ctx.completion_text
        )

        ollama_url = getattr(config, "ollama_base_url", "http://localhost:11434/api/generate")
        model_name = getattr(config, "ollama_model_name", "batiai/gemma4-e2b:q4")
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json" # Ollama supports forcing JSON format
        }

        try:
            client = self._get_client()
            resp = await client.post(ollama_url, json=payload)
            resp.raise_for_status()
            result_data = resp.json()
            judge_output = result_data.get("response", "")
            
            # Parse the SLM judge's JSON output
            try:
                parsed = json.loads(judge_output)
                is_hallucination = parsed.get("is_hallucination", False)
                reason = parsed.get("reason", "No reason provided")
            except json.JSONDecodeError:
                logger.warning("Trace %s: SLM Judge output invalid JSON: %s", ctx.trace_id, judge_output)
                # Fail-open if the judge model failed to format properly
                return self._allow(metadata={"reason": "judge_output_parse_error", "raw_output": judge_output})

        except Exception as e:
            logger.error("Trace %s: Ollama API call failed: %s", ctx.trace_id, e)
            # Fail-open on API errors to avoid breaking the proxy if Ollama is down
            return self._allow(metadata={"reason": "ollama_api_error", "error": str(e)})

        metadata = {
            "judge_model": model_name,
            "is_hallucination": is_hallucination,
            "judge_reason": reason,
        }

        if is_hallucination:
            logger.warning("Trace %s: Chatbot hallucination detected by SLM judge. Reason: %s", ctx.trace_id, reason)
            action_val = getattr(config, "action_on_violation", "rewrite")
            action = PDPAction.REWRITE if action_val == "rewrite" else PDPAction.BLOCK
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=False,
                suggested_action=action,
                violation_code=ViolationCode.HALLUCINATION,
                reason=f"SLM Judge detected hallucination: {reason}",
                metadata=metadata,
            )

        logger.debug("Trace %s: Chatbot hallucination check PASSED.", ctx.trace_id)
        return self._allow(metadata=metadata)

    def _allow(self, score: Optional[float] = None, metadata: Optional[Dict] = None) -> DetectionResult:
        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            score=score,
            suggested_action=PDPAction.ALLOW,
            metadata=metadata or {},
        )
