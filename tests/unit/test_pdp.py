import pytest
from controlplane.detectors.base import DetectionResult, GuardStage, GuardContext
from controlplane.policy.models import PolicyDefinition, PostExecutionConfig
from controlplane.pdp.engine import PDPEngine
from controlplane.pdp.types import PDPAction, ViolationCode


def test_pdp_decision_priority():
    pdp = PDPEngine()
    policy = PolicyDefinition(
        post_execution=PostExecutionConfig(fallback_message="Fallback message.")
    )
    ctx = GuardContext(trace_id="trace_pdp_1", app_id="test", policy=policy)

    # 1. If any result is BLOCK -> Decision is BLOCK
    res_block = DetectionResult(
        detector_name="inj",
        stage=GuardStage.PRE_EXECUTION,
        passed=False,
        suggested_action=PDPAction.BLOCK,
        violation_code=ViolationCode.PROMPT_INJECTION,
    )
    res_rewrite = DetectionResult(
        detector_name="rag",
        stage=GuardStage.POST_EXECUTION,
        passed=False,
        suggested_action=PDPAction.REWRITE,
        violation_code=ViolationCode.HALLUCINATION,
    )
    decision = pdp.evaluate(ctx, [res_block, res_rewrite])
    assert decision.action == PDPAction.BLOCK
    assert decision.http_status == 422

    # 2. If no BLOCK but REWRITE exists -> Decision is REWRITE
    decision_rewrite = pdp.evaluate(ctx, [res_rewrite])
    assert decision_rewrite.action == PDPAction.REWRITE
    assert decision_rewrite.http_status == 200
    assert decision_rewrite.fallback_payload is not None
    assert "Fallback message." in decision_rewrite.fallback_payload["choices"][0]["message"]["content"]
