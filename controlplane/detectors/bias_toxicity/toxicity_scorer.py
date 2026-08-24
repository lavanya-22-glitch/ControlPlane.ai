import re
import logging
from typing import List, Tuple

from controlplane.detectors.base import BaseDetector, GuardStage, DetectionResult, GuardContext
from controlplane.pdp.types import PDPAction, ViolationCode

logger = logging.getLogger("controlplane.detectors.toxicity")

# High-precision patterns for toxicity, harassment, and severe profanity
TOXIC_PATTERNS: List[Tuple[re.Pattern, float, str]] = [
    (re.compile(r"\b(?:kill\s+yourself|go\s+die|commit\s+suicide)\b", re.IGNORECASE), 1.0, "Self-harm incitement"),
    (re.compile(r"\b(?:hate|despise|destroy)\s+all\s+(?:immigrants|refugees|minorities|women|men)\b", re.IGNORECASE), 0.95, "Identity-targeted hate speech"),
    (re.compile(r"\b(?:shut\s+the\s+fuck\s+up|you\s+piece\s+of\s+shit|dumbass|idiot)\b", re.IGNORECASE), 0.85, "Abusive/toxic language"),
]


class ToxicityGuard(BaseDetector):
    """Fast Toxicity and Harmful Content Classifier."""

    @property
    def name(self) -> str:
        return "toxicity_classifier"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.POST_EXECUTION

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        if not ctx.policy.post_execution or ctx.policy.post_execution.toxicity_threshold is None:
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.ALLOW,
            )

        threshold = ctx.policy.post_execution.toxicity_threshold
        text = ctx.completion_text or ""
        if not text:
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.ALLOW,
            )

        max_score = 0.0
        reasons = []

        for pattern, score, reason in TOXIC_PATTERNS:
            if pattern.search(text):
                max_score = max(max_score, score)
                reasons.append(reason)

        passed = max_score < threshold
        if not passed:
            action = (
                PDPAction.REWRITE
                if ctx.policy.post_execution.action_on_violation == "rewrite"
                else PDPAction.BLOCK
            )
            reason_str = f"Toxicity score {max_score:.2f} exceeded threshold {threshold:.2f} ({', '.join(reasons)})"
            logger.warning("Trace %s: %s", ctx.trace_id, reason_str)
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=False,
                score=max_score,
                suggested_action=action,
                violation_code=ViolationCode.TOXICITY,
                reason=reason_str,
                metadata={"matched_toxic_rules": reasons},
            )

        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            score=max_score,
            suggested_action=PDPAction.ALLOW,
        )
