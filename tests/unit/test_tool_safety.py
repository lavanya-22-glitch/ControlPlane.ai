import json
import pytest
from controlplane.detectors.tool_safety.validator import AgentToolGuard
from controlplane.detectors.base import GuardContext
from controlplane.policy.models import PolicyDefinition, ToolGuard, ParameterRule
from controlplane.pdp.types import PDPAction, ViolationCode


@pytest.mark.asyncio
async def test_tool_safety_bounds_and_allowlist():
    guard = AgentToolGuard()
    policy = PolicyDefinition(
        mode="agent",
        tool_guards={
            "execute_command": ToolGuard(
                parameter_rules=[
                    ParameterRule(field="command_type", type="string", allowed_values=["READ", "QUERY"]),
                    ParameterRule(field="timeout_seconds", type="int", min_value=1, max_value=60),
                ]
            )
        },
    )

    # Valid tool parameters
    ctx_valid = GuardContext(
        trace_id="trace_tool_1",
        app_id="agent",
        policy=policy,
        tool_calls=[
            {
                "function": {
                    "name": "execute_command",
                    "arguments": json.dumps({"command_type": "QUERY", "timeout_seconds": 30}),
                }
            }
        ],
    )
    res_valid = await guard.evaluate(ctx_valid)
    assert res_valid.passed is True
    assert res_valid.suggested_action == PDPAction.ALLOW

    # Out of bounds timeout
    ctx_oob = GuardContext(
        trace_id="trace_tool_2",
        app_id="agent",
        policy=policy,
        tool_calls=[
            {
                "function": {
                    "name": "execute_command",
                    "arguments": json.dumps({"command_type": "QUERY", "timeout_seconds": 9999}),
                }
            }
        ],
    )
    res_oob = await guard.evaluate(ctx_oob)
    assert res_oob.passed is False
    assert res_oob.suggested_action == PDPAction.BLOCK
    assert res_oob.violation_code == ViolationCode.TOOL_OUT_OF_BOUNDS

    # Disallowed command_type
    ctx_disallowed = GuardContext(
        trace_id="trace_tool_3",
        app_id="agent",
        policy=policy,
        tool_calls=[
            {
                "function": {
                    "name": "execute_command",
                    "arguments": json.dumps({"command_type": "DROP_DATABASE", "timeout_seconds": 10}),
                }
            }
        ],
    )
    res_disallowed = await guard.evaluate(ctx_disallowed)
    assert res_disallowed.passed is False
    assert res_disallowed.suggested_action == PDPAction.BLOCK
    assert res_disallowed.violation_code == ViolationCode.TOOL_VALUE_DISALLOWED
