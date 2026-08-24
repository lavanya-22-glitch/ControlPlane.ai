from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ParameterRule(BaseModel):
    field: str
    type: str  # "string", "int", "float", "bool", "list", "dict"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None


class ToolGuard(BaseModel):
    parameter_rules: List[ParameterRule] = Field(default_factory=list)
    action_on_violation: str = "block"  # "block" | "rewrite"


class PreExecutionConfig(BaseModel):
    block_prompt_injection: bool = True
    injection_threshold: float = 0.60
    pii_action: str = "mask"  # "mask" | "block" | "none"
    masked_entities: List[str] = Field(
        default_factory=lambda: ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "SSN"]
    )


class HallucinationConfig(BaseModel):
    enabled: bool = True
    min_grounding_score: float = 0.70
    action_on_low_grounding: str = "rewrite"
    fallback_message: str = (
        "The requested details could not be verified against source documents."
    )


class PostExecutionConfig(BaseModel):
    toxicity_threshold: Optional[float] = 0.70
    bias_subspace_threshold: Optional[float] = 0.65
    hallucination_engine: Optional[HallucinationConfig] = None
    action_on_violation: str = "rewrite"
    fallback_message: Optional[str] = (
        "I am unable to answer this query due to safety policy constraints."
    )


class PolicyDefinition(BaseModel):
    version: str = "1.0.0"
    mode: str = "chatbot"  # "chatbot" | "rag" | "agent"
    fail_mode: str = "fail_closed"  # "fail_closed" | "fail_open"
    streaming_mode: str = "buffered"  # "buffered" | "pass_through" | "deny"
    pre_execution: PreExecutionConfig = Field(default_factory=PreExecutionConfig)
    post_execution: Optional[PostExecutionConfig] = Field(
        default_factory=PostExecutionConfig
    )
    tool_guards: Optional[Dict[str, ToolGuard]] = None
    policy_hash: Optional[str] = None


class PoliciesFile(BaseModel):
    policies: Dict[str, PolicyDefinition]
