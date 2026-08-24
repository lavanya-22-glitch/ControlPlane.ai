import time
from typing import Dict, Any, Optional
from controlplane.pdp.types import ViolationCode, PDPAction


def build_error_response(
    code: ViolationCode,
    message: str,
    trace_id: str,
    policy_version: str,
    action: PDPAction = PDPAction.BLOCK,
) -> Dict[str, Any]:
    """Construct a wire-compatible structured 422 error body."""
    return {
      "error": {
        "type": "controlplane_policy_violation",
        "code": code.value,
        "message": message,
        "trace_id": trace_id,
        "policy_version": policy_version,
        "pdp_action": action.value,
      }
    }


def build_fallback_completion(
    fallback_text: str,
    trace_id: str,
    model_name: str = "controlplane-safe-fallback",
) -> Dict[str, Any]:
    """Construct an OpenAI wire-compatible completion payload with the safe fallback."""
    token_count = max(1, len(fallback_text.split()))
    return {
        "id": f"chatcmpl-safe-{trace_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": fallback_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": token_count,
            "total_tokens": token_count,
        },
        "controlplane_meta": {
            "pdp_action": "REWRITE",
            "rewritten": True,
        }
    }
