import pytest
from controlplane.detectors.hallucination.nli_verifier import NLIRAGGroundingGuard
from controlplane.detectors.base import GuardContext
from controlplane.policy.models import PolicyDefinition, PostExecutionConfig, HallucinationConfig
from controlplane.pdp.types import PDPAction, ViolationCode


@pytest.mark.asyncio
async def test_hallucination_detection():
    guard = NLIRAGGroundingGuard()
    policy = PolicyDefinition(
        post_execution=PostExecutionConfig(
            hallucination_engine=HallucinationConfig(
                enabled=True,
                min_grounding_score=0.70,
                action_on_low_grounding="rewrite",
                fallback_message="Unverified answer.",
            )
        )
    )

    # Grounded response matching context
    ctx_grounded = GuardContext(
        trace_id="trace_rag_1",
        app_id="test",
        policy=policy,
        retrieved_context=["Refunds are issued within 5 to 7 business days."],
        completion_text="Refunds are typically processed within 5 to 7 business days.",
    )
    res_grounded = await guard.evaluate(ctx_grounded)
    assert res_grounded.passed is True
    assert res_grounded.suggested_action == PDPAction.ALLOW

    # Hallucinated response contradicting or ignoring context
    ctx_hallucinated = GuardContext(
        trace_id="trace_rag_2",
        app_id="test",
        policy=policy,
        retrieved_context=["Refunds are issued within 5 to 7 business days."],
        completion_text="All customers receive 200% airline vouchers valid for 100 years immediately.",
    )
    res_hallucinated = await guard.evaluate(ctx_hallucinated)
    assert res_hallucinated.passed is False
    assert res_hallucinated.suggested_action == PDPAction.REWRITE
    assert res_hallucinated.violation_code == ViolationCode.HALLUCINATION
