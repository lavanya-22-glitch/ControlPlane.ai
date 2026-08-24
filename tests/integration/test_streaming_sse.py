import json
import pytest
from httpx import AsyncClient, ASGITransport

from controlplane.main import app
from controlplane.policy.registry import policy_registry
from controlplane.telemetry.sink import telemetry_sink
from controlplane.telemetry.query import telemetry_query


@pytest.mark.asyncio
async def test_streaming_sse_pipeline(temp_policy_file, temp_db):
    policy_registry._config_path = temp_policy_file
    policy_registry.reload()
    telemetry_sink.db_path = temp_db
    telemetry_query.db_path = temp_db
    await telemetry_sink.initialize()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Request with stream=true
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Update my email to test@domain.com"}
            ],
            "stream": True,
        }

        response = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"X-ControlPlane-App-ID": "customer-support"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Read chunks from the SSE stream
        body = response.text
        lines = [line.strip() for line in body.split("\n") if line.strip().startswith("data: ")]
        assert len(lines) > 0
        assert lines[-1] == "data: [DONE]"

        # Check token chunk format
        first_data = json.loads(lines[0][6:])
        assert "choices" in first_data
        assert "delta" in first_data["choices"][0]
