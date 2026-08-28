import logging
import hashlib
import json
from typing import Dict, Any

from controlplane.detectors.base import BaseDetector, GuardStage, DetectionResult, GuardContext
from controlplane.pdp.types import PDPAction, ViolationCode
from controlplane.policy.models import AgentConfig

logger = logging.getLogger("controlplane.detectors.token_optim.tool_deduplicator")

def _hash_tool_call(name: str, arguments: str) -> str:
    """Creates a deterministic hash for a tool call to identify duplicates."""
    # Strip whitespace from JSON arguments to ensure consistent hashing
    try:
        parsed_args = json.loads(arguments)
        canonical_args = json.dumps(parsed_args, sort_keys=True)
    except Exception:
        canonical_args = arguments
        
    raw = f"{name}::{canonical_args}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def _get_agent_config(ctx: GuardContext) -> AgentConfig:
    return getattr(ctx.policy, "agent_config", None)

class ToolDeduplicationGuard(BaseDetector):
    """
    Prevents agents from getting stuck in infinite loops by repeatedly calling 
    the same tool with the exact same arguments. 
    If a duplicate is detected, it short-circuits the tool call and injects the cached output.
    """

    @property
    def name(self) -> str:
        return "tool_deduplicator"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.POST_EXECUTION

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        agent_config = _get_agent_config(ctx)
        if not agent_config or not getattr(agent_config, "tool_deduplication", None):
            return self._allow()
            
        dedup_config = agent_config.tool_deduplication
        if not dedup_config.enabled:
            return self._allow()

        if not ctx.tool_calls:
            return self._allow()

        # 1. Build a hash map of all historically completed tool calls in this trace
        # Map: tool_hash -> {"count": int, "output": str}
        history_cache = {}
        
        # We need to map tool_call_id -> hash to link the tool responses
        id_to_hash = {}
        
        for msg in ctx.messages:
            role = msg.get("role", "")
            
            # Map assistant's tool calls to hashes
            if role == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id")
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", "")
                    tc_hash = _hash_tool_call(name, args)
                    if tc_id:
                        id_to_hash[tc_id] = tc_hash
                        
            # Map tool responses to the hash
            elif role == "tool":
                tc_id = msg.get("tool_call_id")
                content = msg.get("content", "")
                if tc_id in id_to_hash:
                    tc_hash = id_to_hash[tc_id]
                    if tc_hash not in history_cache:
                        history_cache[tc_hash] = {"count": 0, "output": content}
                    history_cache[tc_hash]["count"] += 1
                    # Keep latest output
                    history_cache[tc_hash]["output"] = content

        # 2. Check current requested tool calls against the cache
        for tc in ctx.tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", "")
            tc_hash = _hash_tool_call(name, args)
            
            if tc_hash in history_cache:
                count = history_cache[tc_hash]["count"]
                
                # The LLM has requested this before, and received an answer `count` times.
                # If they are requesting it AGAIN (e.g. for the 2nd or 3rd time):
                if count >= dedup_config.max_duplicate_calls:
                    cached_output = history_cache[tc_hash]["output"]
                    reason = f"Agent attempted to call '{name}' with identical arguments {count + 1} times."
                    logger.warning(f"Trace {ctx.trace_id}: {reason} Short-circuiting.")
                    
                    # We create a fallback payload that strips the tool calls and forces the agent to read the cache.
                    fallback_text = (
                        f"[ControlPlane System Intercept]: You attempted to call '{name}' with identical arguments "
                        f"that you have already executed. To save tokens and prevent infinite loops, the tool call was blocked. "
                        f"\n\nHere is the cached output from your previous call:\n{cached_output}\n\n"
                        f"Please read the above output carefully and proceed with your reasoning."
                    )
                    
                    # If there was a <thinking> block, preserve it!
                    if ctx.completion_text:
                        fallback_text = ctx.completion_text + "\n\n" + fallback_text
                        
                    fallback_payload = {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": fallback_text,
                                    # We remove the tool_calls entirely!
                                }
                            }
                        ]
                    }
                    
                    return DetectionResult(
                        detector_name=self.name,
                        stage=self.stage,
                        passed=False,
                        suggested_action=PDPAction.REWRITE,
                        violation_code=ViolationCode.DUPLICATE_TOOL_CALL,
                        reason=reason,
                        fallback_payload=fallback_payload,
                        metadata={"tool_name": name, "duplicate_count": count + 1}
                    )

        return self._allow()

    def _allow(self) -> DetectionResult:
        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            suggested_action=PDPAction.ALLOW,
        )
