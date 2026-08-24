import asyncio
import logging
from pathlib import Path
from typing import Optional
import aiosqlite

from controlplane.telemetry.schema import CREATE_TABLE_SQL, AuditTraceRecord

logger = logging.getLogger("controlplane.telemetry.sink")

INSERT_TRACE_SQL = """
INSERT OR REPLACE INTO audit_traces (
    trace_id, app_id, policy_version, policy_hash, use_case_mode,
    input_tokens, output_tokens, latency_ms, pre_execution_latency_ms,
    upstream_latency_ms, post_execution_latency_ms, pii_entities_found,
    injection_score, grounding_score, bias_score, pdp_action,
    violation_reason, raw_request_payload, final_response_payload
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


class TelemetrySink:
    """Async SQLite Write-Ahead Logging (WAL) audit sink."""

    def __init__(self, db_path: str | Path = "data/audit_logs.db"):
        self.db_path = Path(db_path)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create database directory, enable WAL mode, and ensure schema exists."""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA synchronous=NORMAL;")
            await db.executescript(CREATE_TABLE_SQL)
            await db.commit()
        self._initialized = True
        logger.info("Telemetry sink initialized at %s (WAL mode active)", self.db_path)

    async def record_trace(self, record: AuditTraceRecord) -> None:
        """Persist an execution trace asynchronously."""
        if not self._initialized:
            await self.initialize()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(INSERT_TRACE_SQL, record.to_sqlite_tuple())
                await db.commit()
            logger.debug("Persisted trace %s to audit sink.", record.trace_id)
        except Exception as e:
            logger.error("Failed to write trace %s to audit sink: %s", record.trace_id, e)


# Global telemetry sink singleton
telemetry_sink = TelemetrySink()
