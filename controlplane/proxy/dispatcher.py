import os
import logging
from typing import Dict, Any, Optional
import httpx

from controlplane.mock_upstream.server import mock_llm_provider

logger = logging.getLogger("controlplane.proxy.dispatcher")


class UpstreamDispatcher:
    """High-concurrency async dispatcher for upstream LLMs."""

    def __init__(self):
        self.upstream_url = os.getenv("UPSTREAM_URL", "https://api.openai.com/v1/chat/completions")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.mock_mode = os.getenv("CONTROLPLANE_MOCK_MODE", "true").lower() == "true"
        self._client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
            )
        return self._client

    async def dispatch(self, payload: Dict[str, Any], raw_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Dispatch payload to upstream LLM API or fallback to internal mock provider."""
        # Use mock provider if explicitly configured or API key is absent
        if self.mock_mode or not self.api_key:
            return await mock_llm_provider.complete(payload)

        client = await self.get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if raw_headers and "authorization" in raw_headers:
            headers["Authorization"] = raw_headers["authorization"]

        # Clean custom ControlPlane fields before forwarding upstream
        clean_payload = {k: v for k, v in payload.items() if k != "retrieved_context"}

        response = await client.post(self.upstream_url, json=clean_payload, headers=headers)
        
        if not response.is_success:
            # Forward the upstream error body and status code as-is so the
            # client sees the real error (e.g., 429 quota, 404 model not found)
            # instead of a generic 500 from our proxy crashing.
            from fastapi.responses import JSONResponse
            logger.warning(
                "Upstream returned %s: %s",
                response.status_code,
                response.text[:200],
            )
            raise httpx.HTTPStatusError(
                message=response.text,
                request=response.request,
                response=response,
            )
        
        return response.json()

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


dispatcher = UpstreamDispatcher()
