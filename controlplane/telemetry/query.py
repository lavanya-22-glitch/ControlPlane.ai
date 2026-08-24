import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import aiosqlite

logger = logging.getLogger("controlplane.telemetry.query")


class TelemetryQueryService:
    """Read-only service for inspecting audit traces and computing live metrics."""

    def __init__(self, db_path: str | Path = "data/audit_logs.db"):
        self.db_path = Path(db_path)

    async def get_recent_traces(
        self, limit: int = 50, app_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self.db_path = Path(self.db_path)
        if not self.db_path.exists():
            return []

        query = "SELECT * FROM audit_traces"
        params = []
        if app_id:
            query += " WHERE app_id = ?"
            params.append(app_id)
        query += " ORDER BY timestamp DESC LIMIT ?;"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    item = dict(row)
                    if item.get("pii_entities_found"):
                        try:
                            item["pii_entities_found"] = json.loads(item["pii_entities_found"])
                        except Exception:
                            item["pii_entities_found"] = []
                    results.append(item)
                return results

    async def get_metrics_summary(self) -> Dict[str, Any]:
        self.db_path = Path(self.db_path)
        if not self.db_path.exists():
            return {
                "total_requests": 0,
                "allow_count": 0,
                "transform_count": 0,
                "rewrite_count": 0,
                "block_count": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
            }

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN pdp_action = 'ALLOW' THEN 1 ELSE 0 END) as allows,
                    SUM(CASE WHEN pdp_action = 'TRANSFORM' THEN 1 ELSE 0 END) as transforms,
                    SUM(CASE WHEN pdp_action = 'REWRITE' THEN 1 ELSE 0 END) as rewrites,
                    SUM(CASE WHEN pdp_action = 'BLOCK' THEN 1 ELSE 0 END) as blocks,
                    AVG(latency_ms) as avg_latency
                FROM audit_traces;
                """
            ) as cursor:
                row = await cursor.fetchone()
                if not row or row["total"] == 0:
                    return {
                        "total_requests": 0,
                        "allow_count": 0,
                        "transform_count": 0,
                        "rewrite_count": 0,
                        "block_count": 0,
                        "avg_latency_ms": 0.0,
                    }

                return {
                    "total_requests": row["total"] or 0,
                    "allow_count": row["allows"] or 0,
                    "transform_count": row["transforms"] or 0,
                    "rewrite_count": row["rewrites"] or 0,
                    "block_count": row["blocks"] or 0,
                    "avg_latency_ms": round(row["avg_latency"] or 0.0, 2),
                }


telemetry_query = TelemetryQueryService()
