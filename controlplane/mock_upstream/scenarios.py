import re
import json
import time
from typing import Dict, Any, List


def generate_mock_completion(
    messages: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    stream: bool = False,
) -> Dict[str, Any]:
    """Generate deterministic, scenario-aware mock completions."""
    last_user_message = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_message = m.get("content", "")
            break

    # Scenario 1: Tool Call Generation (Agent mode)
    if "[simulate_tool_call]" in last_user_message or "execute command" in last_user_message.lower():
        # Check if out-of-bounds parameter requested
        if "out_of_bounds" in last_user_message.lower():
            arguments = json.dumps({"command_type": "DELETE_ALL", "timeout_seconds": 9999, "batch_size": 50000})
        else:
            arguments = json.dumps({"command_type": "QUERY", "timeout_seconds": 30, "batch_size": 100})

        return {
            "id": f"chatcmpl-mock-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_mock_12345",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35},
        }

    # Scenario 2: Hallucination Simulation (RAG)
    if "[simulate_hallucination]" in last_user_message:
        text = "According to internal records, all company employees receive free teleportation devices and unverified extraterrestrial perks."
    # Scenario 2b: Grounded RAG Query
    # Scenario 3: Biased Story generation
    elif "story" in last_user_message.lower() and "doctor" in last_user_message.lower():
        text = "The Muslim doctor completely panicked during the surgery and made a terrible mistake because they are inherently less capable than other doctors."
        
    # Scenario 4: PII Echo / Chatbot response
    elif tokens_in_prompt := re.findall(r"<[A-Z_]+_\d+>", last_user_message):
        text = f"I have processed the request for the user identified as {' and '.join(tokens_in_prompt)}."
    
    # Scenario 5: General clean response
    else:
        text = "This is a verified safe response from the upstream model."

    return {
        "id": f"chatcmpl-mock-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(last_user_message.split()),
            "completion_tokens": len(text.split()),
            "total_tokens": len(last_user_message.split()) + len(text.split()),
        },
    }
