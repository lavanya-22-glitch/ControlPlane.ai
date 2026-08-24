import json
import asyncio
from typing import Dict, Any, AsyncGenerator
from controlplane.mock_upstream.scenarios import generate_mock_completion


class MockLLMProvider:
    """In-memory or standalone mock provider matching OpenAI /v1/chat/completions wire format."""

    async def complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute synchronous mock chat completion."""
        messages = payload.get("messages", [])
        model = payload.get("model", "gpt-4o-mini")
        # Add micro-sleep to simulate realistic LLM latency (20ms)
        await asyncio.sleep(0.02)
        return generate_mock_completion(messages=messages, model=model, stream=False)

    async def stream_complete(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Execute Server-Sent Events (SSE) streaming mock completion."""
        full_response = await self.complete(payload)
        choice = full_response["choices"][0]
        content = choice.get("message", {}).get("content")

        if content:
            words = content.split(" ")
            for i, word in enumerate(words):
                chunk = {
                    "id": full_response["id"],
                    "object": "chat.completion.chunk",
                    "created": full_response["created"],
                    "model": full_response["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": word + (" " if i < len(words) - 1 else "")},
                            "finish_reason": None if i < len(words) - 1 else "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)
        yield "data: [DONE]\n\n"


# Global mock provider singleton
mock_llm_provider = MockLLMProvider()
