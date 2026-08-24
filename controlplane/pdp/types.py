from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class PDPAction(str, Enum):
    ALLOW = "ALLOW"
    TRANSFORM = "TRANSFORM"
    REWRITE = "REWRITE"
    BLOCK = "BLOCK"


class ViolationCode(str, Enum):
    NONE = "NONE"
    PROMPT_INJECTION = "PROMPT_INJECTION_DETECTED"
    PII_DISALLOWED = "PII_POLICY_VIOLATION"
    HALLUCINATION = "LOW_GROUNDING_HALLUCINATION"
    TOXICITY = "TOXIC_CONTENT_DETECTED"
    BIAS = "BIAS_SUBSPACE_DRIFT"
    TOOL_TYPE_MISMATCH = "TOOL_PARAMETER_TYPE_MISMATCH"
    TOOL_OUT_OF_BOUNDS = "TOOL_PARAMETER_OUT_OF_BOUNDS"
    TOOL_VALUE_DISALLOWED = "TOOL_PARAMETER_VALUE_DISALLOWED"
    DETECTOR_ERROR = "INTERNAL_DETECTOR_EXCEPTION"


class PDPDecision(BaseModel):
    action: PDPAction
    violation_code: ViolationCode = ViolationCode.NONE
    reason: Optional[str] = None
    http_status: int = 200
    fallback_payload: Optional[Dict[str, Any]] = None
    transformed_payload: Optional[Dict[str, Any]] = None
    scores: Dict[str, Optional[float]] = {}
    entities_found: List[str] = []
    metadata: Dict[str, Any] = {}
