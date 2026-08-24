import json
import time
import logging
from typing import AsyncGenerator, Dict, Any, List

from controlplane.detectors.base import GuardContext
from controlplane.detectors.pii.detokenizer import PIIDetokenizerGuard
from controlplane.pdp.engine import pdp_engine
from controlplane.pdp.types import PDPAction

logger = logging.getLogger("controlplane.proxy.stream")


async def handle_buffered_stream(
    upstream_generator: AsyncGenerator[str, None],
    ctx: GuardContext,
    post_guards: List[Any],
    model: str = "gpt-4o-mini",
) -> AsyncGenerator[str, None]:
    """Buffer stream chunks, execute post-execution guards on complete text, and flush."""
    aggregated_content = []
    chunk_id = f"chatcmpl-stream-{ctx.trace_id}"

    # 1. Accumulate stream
    async for raw_line in upstream_generator:
        if not raw_line.startswith("data: "):
            continue
        data_str = raw_line[6:].strip()
        if data_str == "[DONE]":
            break

        try:
            parsed = json.loads(data_str)
            delta = parsed["choices"][0].get("delta", {})
            if "content" in delta and delta["content"]:
                aggregated_content.append(delta["content"])
        except Exception:
            continue

    full_text = "".join(aggregated_content)
    ctx.completion_text = full_text

    # 2. Run post-execution guards
    results = []
    for guard in post_guards:
        try:
            res = await guard.evaluate(ctx)
            results.append(res)
        except Exception as e:
            logger.error("Error in stream post guard %s: %s", guard.name, e)

    # 3. PDP Evaluation
    decision = pdp_engine.evaluate(ctx, results, upstream_model=model)

    # 4. Stream response to client based on PDP decision
    if decision.action == PDPAction.BLOCK:
        err_chunk = {
            "error": {
                "type": "controlplane_policy_violation",
                "code": decision.violation_code.value,
                "message": decision.reason,
                "trace_id": ctx.trace_id,
            }
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    output_text = ctx.completion_text or ""
    if decision.action == PDPAction.REWRITE and decision.fallback_payload:
        output_text = decision.fallback_payload["choices"][0]["message"]["content"]

    # Stream out the verified & sanitized tokens
    words = output_text.split(" ")
    for i, word in enumerate(words):
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word + (" " if i < len(words) - 1 else "")},
                    "finish_reason": None if i < len(words) - 1 else "stop",
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"
