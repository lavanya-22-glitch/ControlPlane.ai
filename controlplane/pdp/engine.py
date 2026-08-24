from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from controlplane.detectors.base import DetectionResult, GuardContext

from controlplane.pdp.actions import build_error_response, build_fallback_completion
from controlplane.pdp.types import PDPAction, PDPDecision, ViolationCode
from controlplane.policy.models import PolicyDefinition

logger = logging.getLogger("controlplane.pdp")


class PDPEngine:
    """Deterministic Policy Decision Point Engine."""

    def evaluate(
        self,
        ctx: GuardContext,
        results: List[DetectionResult],
        upstream_model: str = "gpt-4o",
    ) -> PDPDecision:
        """Map detection outputs and active policy rules to a final PDP decision."""
        policy = ctx.policy
        scores: Dict[str, Optional[float]] = {}
        entities_found: List[str] = list(ctx.pii_entities_detected)

        # Collect scores and entities
        for res in results:
            if res.score is not None:
                scores[res.detector_name] = res.score
            if "entities" in res.metadata:
                entities_found.extend(res.metadata["entities"])

        # Check for BLOCK actions first
        block_violations = [
            r for r in results if not r.passed and r.suggested_action == PDPAction.BLOCK
        ]
        if block_violations:
            primary = block_violations[0]
            logger.warning(
                "PDP decision BLOCK triggered for trace %s: %s (%s)",
                ctx.trace_id,
                primary.violation_code,
                primary.reason,
            )
            return PDPDecision(
                action=PDPAction.BLOCK,
                violation_code=primary.violation_code,
                reason=primary.reason or "Policy violation detected.",
                http_status=422,
                scores=scores,
                entities_found=list(set(entities_found)),
                metadata={"detector": primary.detector_name, **primary.metadata},
            )

        # Check for REWRITE actions (e.g. Hallucination or Content Toxicity rewrite)
        rewrite_violations = [
            r for r in results if not r.passed and r.suggested_action == PDPAction.REWRITE
        ]
        if rewrite_violations:
            primary = rewrite_violations[0]
            fallback_text = (
                primary.metadata.get("fallback_message")
                or (
                    policy.post_execution.fallback_message
                    if policy.post_execution
                    else None
                )
                or "The response could not be verified and was replaced by safety policy."
            )
            fallback_payload = build_fallback_completion(
                fallback_text=fallback_text,
                trace_id=ctx.trace_id,
                model_name=upstream_model,
            )
            logger.info(
                "PDP decision REWRITE triggered for trace %s: %s",
                ctx.trace_id,
                primary.violation_code,
            )
            return PDPDecision(
                action=PDPAction.REWRITE,
                violation_code=primary.violation_code,
                reason=primary.reason,
                http_status=200,
                fallback_payload=fallback_payload,
                scores=scores,
                entities_found=list(set(entities_found)),
                metadata={"detector": primary.detector_name, **primary.metadata},
            )

        # Check for TRANSFORM action (e.g. PII masking)
        transform_triggers = [
            r for r in results if r.suggested_action == PDPAction.TRANSFORM
        ]
        if transform_triggers:
            return PDPDecision(
                action=PDPAction.TRANSFORM,
                violation_code=ViolationCode.NONE,
                reason="Payload sanitized or transformed.",
                http_status=200,
                scores=scores,
                entities_found=list(set(entities_found)),
            )

        # Default ALLOW
        return PDPDecision(
            action=PDPAction.ALLOW,
            violation_code=ViolationCode.NONE,
            reason="All guard checks passed.",
            http_status=200,
            scores=scores,
            entities_found=list(set(entities_found)),
        )


# Global PDP engine singleton
pdp_engine = PDPEngine()
