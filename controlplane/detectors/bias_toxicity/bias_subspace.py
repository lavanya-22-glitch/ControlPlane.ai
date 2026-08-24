import re
import logging
from typing import List, Tuple

from controlplane.detectors.base import BaseDetector, GuardStage, DetectionResult, GuardContext
from controlplane.pdp.types import PDPAction, ViolationCode

logger = logging.getLogger("controlplane.detectors.bias")

BIAS_STEREOTYPE_PATTERNS: List[Tuple[re.Pattern, float, str]] = [
    (re.compile(r"\b(?:all|every)\s+(?:women|men|asians|black\s+people|hispanics|white\s+people)\s+are\s+(?:bad|lazy|dumb|greedy|violent|emotional)\b", re.IGNORECASE), 0.90, "Generalizing demographic stereotype"),
]


class BiasSubspaceGuard(BaseDetector):
    """Demographic Bias and Subspace Skew Detector."""

    @property
    def name(self) -> str:
        return "bias_subspace_detector"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.POST_EXECUTION

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        if not ctx.policy.post_execution or ctx.policy.post_execution.bias_subspace_threshold is None:
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.ALLOW,
            )

        threshold = ctx.policy.post_execution.bias_subspace_threshold
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
        for pattern, score, reason in BIAS_STEREOTYPE_PATTERNS:
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
            reason_str = f"Bias subspace drift score {max_score:.2f} exceeded threshold {threshold:.2f} ({', '.join(reasons)})"
            logger.warning("Trace %s: %s", ctx.trace_id, reason_str)
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=False,
                score=max_score,
                suggested_action=action,
                violation_code=ViolationCode.BIAS,
                reason=reason_str,
                metadata={"matched_bias_rules": reasons},
            )

        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            score=max_score,
            suggested_action=PDPAction.ALLOW,
        )
