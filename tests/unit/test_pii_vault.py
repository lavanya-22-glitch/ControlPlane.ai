import pytest
from controlplane.detectors.pii.vault import SessionPIIVault
from controlplane.detectors.pii.scanner import PIIScannerGuard
from controlplane.detectors.pii.detokenizer import PIIDetokenizerGuard
from controlplane.detectors.base import GuardContext
from controlplane.policy.models import PolicyDefinition, PreExecutionConfig
from controlplane.pdp.types import PDPAction


@pytest.mark.asyncio
async def test_pii_vault_token_generation():
    vault = SessionPIIVault()
    trace_id = "test_trace_1"

    token1 = vault.get_or_create_token(trace_id, "EMAIL_ADDRESS", "alice@example.com")
    assert token1 == "<EMAIL_ADDRESS_1>"

    # Same raw value returns same token
    token2 = vault.get_or_create_token(trace_id, "EMAIL_ADDRESS", "alice@example.com")
    assert token2 == "<EMAIL_ADDRESS_1>"

    # Reverse lookup
    assert vault.get_raw_value(trace_id, "<EMAIL_ADDRESS_1>") == "alice@example.com"


@pytest.mark.asyncio
async def test_pii_scanner_masking():
    scanner = PIIScannerGuard()
    policy = PolicyDefinition(
        pre_execution=PreExecutionConfig(
            pii_action="mask",
            masked_entities=["EMAIL_ADDRESS", "CREDIT_CARD"],
        )
    )
    ctx = GuardContext(
        trace_id="trace_pii_test",
        app_id="test",
        policy=policy,
        messages=[{"role": "user", "content": "My email is user@domain.com and card is 4111-2222-3333-4444."}],
    )

    res = await scanner.evaluate(ctx)
    assert res.passed is True
    assert res.suggested_action == PDPAction.TRANSFORM
    assert "<EMAIL_ADDRESS_1>" in ctx.messages[0]["content"]
    assert "<CREDIT_CARD_1>" in ctx.messages[0]["content"]

    # Detokenize
    detokenizer = PIIDetokenizerGuard()
    ctx.completion_text = f"Confirmed email <EMAIL_ADDRESS_1> and card <CREDIT_CARD_1>."
    detok_res = await detokenizer.evaluate(ctx)

    assert detok_res.passed is True
    assert "user@domain.com" in ctx.completion_text
    assert "4111-2222-3333-4444" in ctx.completion_text
