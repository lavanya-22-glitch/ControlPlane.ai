import json
import logging
from typing import Dict, Any, Optional

from controlplane.detectors.base import BaseDetector, GuardStage, DetectionResult, GuardContext
from controlplane.detectors.tool_safety.bounds import check_parameter_bounds
from controlplane.pdp.types import PDPAction, ViolationCode

logger = logging.getLogger("controlplane.detectors.tool_safety")

TYPE_MAP = {
    "string": str,
    "str": str,
    "int": int,
    "integer": int,
    "float": (int, float),
    "number": (int, float),
    "bool": bool,
    "boolean": bool,
    "list": list,
    "array": list,
    "dict": dict,
    "object": dict,
}


class AgentToolGuard(BaseDetector):
    """Interception & Safety Validator for LLM Tool/Function Calling."""

    @property
    def name(self) -> str:
        return "agent_tool_guard"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.TOOL_INTERCEPTION

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        if not ctx.tool_calls or not ctx.policy.tool_guards:
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.ALLOW,
            )

        tool_guards = ctx.policy.tool_guards

        for tool_call in ctx.tool_calls:
            function_obj = tool_call.get("function", {})
            func_name = function_obj.get("name")
            if not func_name or func_name not in tool_guards:
                continue

            guard_def = tool_guards[func_name]
            raw_args = function_obj.get("arguments", "{}")

            # Parse JSON arguments
            if isinstance(raw_args, str):
                try:
                    args_dict = json.loads(raw_args)
                except Exception as e:
                    reason = f"Tool '{func_name}' arguments failed JSON schema parsing: {e}"
                    logger.warning("Trace %s: %s", ctx.trace_id, reason)
                    return DetectionResult(
                        detector_name=self.name,
                        stage=self.stage,
                        passed=False,
                        suggested_action=PDPAction.BLOCK,
                        violation_code=ViolationCode.TOOL_TYPE_MISMATCH,
                        reason=reason,
                    )
            elif isinstance(raw_args, dict):
                args_dict = raw_args
            else:
                args_dict = {}

            # Validate each configured parameter rule
            for rule in guard_def.parameter_rules:
                field_name = rule.field
                if field_name not in args_dict:
                    continue  # Optional or missing field handled if required

                value = args_dict[field_name]
                expected_type = TYPE_MAP.get(rule.type.lower())

                # 1. Type Check
                if expected_type and not isinstance(value, expected_type):
                    reason = (
                        f"Tool '{func_name}' parameter '{field_name}' expected type "
                        f"'{rule.type}', got '{type(value).__name__}'"
                    )
                    logger.warning("Trace %s: %s", ctx.trace_id, reason)
                    return DetectionResult(
                        detector_name=self.name,
                        stage=self.stage,
                        passed=False,
                        suggested_action=PDPAction.BLOCK,
                        violation_code=ViolationCode.TOOL_TYPE_MISMATCH,
                        reason=reason,
                        metadata={"tool_name": func_name, "parameter": field_name},
                    )

                # 2. Bounds & Allowlist Check
                passed_bounds, bounds_error = check_parameter_bounds(rule, value)
                if not passed_bounds:
                    logger.warning("Trace %s: %s", ctx.trace_id, bounds_error)
                    v_code = (
                        ViolationCode.TOOL_VALUE_DISALLOWED
                        if rule.allowed_values and value not in rule.allowed_values
                        else ViolationCode.TOOL_OUT_OF_BOUNDS
                    )
                    return DetectionResult(
                        detector_name=self.name,
                        stage=self.stage,
                        passed=False,
                        suggested_action=PDPAction.BLOCK,
                        violation_code=v_code,
                        reason=bounds_error,
                        metadata={"tool_name": func_name, "parameter": field_name, "value": value},
                    )

        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            suggested_action=PDPAction.ALLOW,
        )
