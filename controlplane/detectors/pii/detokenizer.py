import logging
from typing import Dict, Any, Optional

from controlplane.detectors.base import BaseDetector, GuardStage, DetectionResult, GuardContext
from controlplane.detectors.pii.vault import pii_vault
from controlplane.pdp.types import PDPAction

logger = logging.getLogger("controlplane.detectors.pii.detokenizer")


class PIIDetokenizerGuard(BaseDetector):
    """Restores sensitive tokens on outbound egress payloads."""

    @property
    def name(self) -> str:
        return "pii_detokenizer"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.POST_EXECUTION

    def detokenize_string(self, text: str, trace_id: str) -> str:
        """Replace all known session tokens with their original sensitive values."""
        if not text:
            return text
        mappings = pii_vault.get_all_mappings(trace_id)
        if not mappings:
            return text

        result = text
        for token, raw_val in mappings.items():
            if token in result:
                result = result.replace(token, raw_val)
        return result

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        if not ctx.completion_text:
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.ALLOW,
            )

        detokenized = self.detokenize_string(ctx.completion_text, ctx.trace_id)
        changed = detokenized != ctx.completion_text
        ctx.completion_text = detokenized

        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            suggested_action=PDPAction.TRANSFORM if changed else PDPAction.ALLOW,
            metadata={"detokenized": changed},
        )
