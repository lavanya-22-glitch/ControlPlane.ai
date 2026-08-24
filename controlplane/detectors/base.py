from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from controlplane.pdp.types import PDPAction, ViolationCode
from controlplane.policy.models import PolicyDefinition


class GuardStage(str, Enum):
    PRE_EXECUTION = "PRE_EXECUTION"
    POST_EXECUTION = "POST_EXECUTION"
    TOOL_INTERCEPTION = "TOOL_INTERCEPTION"


class DetectionResult(BaseModel):
    detector_name: str
    stage: GuardStage
    passed: bool
    score: Optional[float] = None
    suggested_action: PDPAction = PDPAction.ALLOW
    violation_code: ViolationCode = ViolationCode.NONE
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GuardContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace_id: str
    app_id: str
    policy: PolicyDefinition
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_context: Optional[List[str]] = None
    completion_text: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    pii_entities_detected: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseDetector(ABC):
    """Abstract interface for all modular safety detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def stage(self) -> GuardStage:
        pass

    @abstractmethod
    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        """Run inspection logic on the given GuardContext."""
        pass
