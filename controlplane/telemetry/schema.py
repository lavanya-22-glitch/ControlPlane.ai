import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_traces (
    trace_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    app_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    use_case_mode TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    latency_ms REAL DEFAULT 0.0,
    pre_execution_latency_ms REAL DEFAULT 0.0,
    upstream_latency_ms REAL DEFAULT 0.0,
    post_execution_latency_ms REAL DEFAULT 0.0,
    pii_entities_found TEXT,
    injection_score REAL,
    grounding_score REAL,
    bias_score REAL,
    pdp_action TEXT NOT NULL,
    violation_reason TEXT,
    raw_request_payload TEXT,
    final_response_payload TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_app_id ON audit_traces (app_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_traces (timestamp);
"""


class AuditTraceRecord(BaseModel):
    trace_id: str
    app_id: str
    policy_version: str = "1.0.0"
    policy_hash: str = "unknown"
    use_case_mode: str = "chatbot"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    pre_execution_latency_ms: float = 0.0
    upstream_latency_ms: float = 0.0
    post_execution_latency_ms: float = 0.0
    pii_entities_found: List[str] = Field(default_factory=list)
    injection_score: Optional[float] = None
    grounding_score: Optional[float] = None
    bias_score: Optional[float] = None
    pdp_action: str = "ALLOW"
    violation_reason: Optional[str] = None
    raw_request_payload: Optional[str] = None
    final_response_payload: Optional[str] = None
    timestamp: Optional[str] = None

    def to_sqlite_tuple(self) -> tuple:
        return (
            self.trace_id,
            self.app_id,
            self.policy_version,
            self.policy_hash,
            self.use_case_mode,
            self.input_tokens,
            self.output_tokens,
            self.latency_ms,
            self.pre_execution_latency_ms,
            self.upstream_latency_ms,
            self.post_execution_latency_ms,
            json.dumps(self.pii_entities_found),
            self.injection_score,
            self.grounding_score,
            self.bias_score,
            self.pdp_action,
            self.violation_reason,
            self.raw_request_payload,
            self.final_response_payload,
        )
