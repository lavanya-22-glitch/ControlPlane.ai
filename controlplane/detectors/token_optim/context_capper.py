import logging
from typing import Dict, Any

from controlplane.detectors.base import BaseDetector, GuardStage, DetectionResult, GuardContext
from controlplane.pdp.types import PDPAction

logger = logging.getLogger("controlplane.detectors.token_optim.context_capper")

class ContextCapperGuard(BaseDetector):
    """Truncates retrieved context in RAG requests to save tokens based on policy limits."""

    @property
    def name(self) -> str:
        return "context_capper"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.PRE_EXECUTION

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        if not ctx.policy.pre_execution:
            return self._allow()
            
        cap_config = getattr(ctx.policy.pre_execution, "context_capping", None)
        if not cap_config or not cap_config.enabled:
            return self._allow()

        if not ctx.retrieved_context:
            return self._allow()

        max_len = cap_config.max_context_length_chars
        current_len = sum(len(chunk) for chunk in ctx.retrieved_context)

        if current_len <= max_len:
            return self._allow()

        # Truncate context list
        truncated_list = []
        running_len = 0
        for chunk in ctx.retrieved_context:
            chunk_len = len(chunk)
            if running_len + chunk_len <= max_len:
                truncated_list.append(chunk)
                running_len += chunk_len
            else:
                # We reached the limit, take whatever fits
                remainder = max_len - running_len
                if remainder > 0:
                    truncated_list.append(chunk[:remainder])
                break
                
        # We append a small indicator that context was truncated for the LLM
        truncation_warning = "\n\n[System: Context truncated due to token optimization limits.]"
        if truncated_list:
            truncated_list[-1] += truncation_warning
        else:
            truncated_list.append(truncation_warning)
            
        ctx.retrieved_context = truncated_list

        logger.info(f"Trace {ctx.trace_id}: Truncated retrieved context from {current_len} to {max_len} chars.")
        
        # We return TRANSFORM because we are modifying the context before it goes to the LLM.
        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            suggested_action=PDPAction.TRANSFORM,
            metadata={"original_length": current_len, "truncated_length": max_len}
        )

    def _allow(self) -> DetectionResult:
        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            suggested_action=PDPAction.ALLOW,
        )
