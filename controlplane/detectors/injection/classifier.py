import logging
from typing import Optional

from controlplane.detectors.base import BaseDetector, GuardStage, DetectionResult, GuardContext
from controlplane.detectors.injection.heuristics import evaluate_heuristics
from controlplane.pdp.types import PDPAction, ViolationCode

logger = logging.getLogger("controlplane.detectors.injection")


class PromptInjectionGuard(BaseDetector):
    """Prompt Injection & Jailbreak Detection Engine."""

    @property
    def name(self) -> str:
        return "prompt_injection"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.PRE_EXECUTION

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        pre_config = ctx.policy.pre_execution
        if not pre_config.block_prompt_injection:
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.ALLOW,
            )

        threshold = pre_config.injection_threshold
        max_score = 0.0
        all_reasons = []

        # Evaluate all user and system messages
        for msg in ctx.messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                score, reasons = evaluate_heuristics(content)
                if score > max_score:
                    max_score = score
                all_reasons.extend(reasons)

        passed = max_score < threshold
        if not passed:
            reason_str = (
                f"Prompt injection risk score {max_score:.2f} exceeded threshold {threshold:.2f}. "
                f"Reasons: {', '.join(set(all_reasons))}"
            )
            logger.warning("Trace %s: %s", ctx.trace_id, reason_str)
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=False,
                score=max_score,
                suggested_action=PDPAction.BLOCK,
                violation_code=ViolationCode.PROMPT_INJECTION,
                reason=reason_str,
                metadata={"matched_rules": list(set(all_reasons))},
            )

        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            score=max_score,
            suggested_action=PDPAction.ALLOW,
        )
