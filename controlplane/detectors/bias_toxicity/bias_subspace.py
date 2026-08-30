import re
import logging
from typing import List, Tuple
import asyncio

from controlplane.detectors.base import BaseDetector, GuardStage, DetectionResult, GuardContext
from controlplane.pdp.types import PDPAction, ViolationCode
from controlplane.detectors.bias_toxicity.bias_projection import BiasProjectionScorer
from controlplane.detectors.bias_toxicity.counterfactual import CounterfactualScorer

logger = logging.getLogger("controlplane.detectors.bias")

BIAS_STEREOTYPE_PATTERNS: List[Tuple[re.Pattern, float, str]] = [
    (re.compile(r"\b(?:all|every)\s+(?:women|men|asians|black\s+people|hispanics|white\s+people)\s+are\s+(?:bad|lazy|dumb|greedy|violent|emotional)\b", re.IGNORECASE), 0.90, "Generalizing demographic stereotype"),
]

class BiasSubspaceGuard(BaseDetector):
    """Demographic Bias and Subspace Skew Detector."""
    
    _projection_scorer = None
    _counterfactual_scorer = None

    def __init__(self):
        if BiasSubspaceGuard._projection_scorer is None:
            BiasSubspaceGuard._projection_scorer = BiasProjectionScorer()
        if BiasSubspaceGuard._counterfactual_scorer is None:
            BiasSubspaceGuard._counterfactual_scorer = CounterfactualScorer()

    @property
    def name(self) -> str:
        return "bias_subspace_detector"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.POST_EXECUTION

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        if not ctx.policy.post_execution or ctx.policy.post_execution.bias_subspace_threshold is None:
            return self._allow()

        config = ctx.policy.post_execution
        threshold = config.bias_subspace_threshold
        text = ctx.completion_text or ""
        if not text:
            return self._allow()

        # 1. Tier-1 Fast Regex Heuristics
        max_score = 0.0
        reasons = []
        for pattern, score, reason in BIAS_STEREOTYPE_PATTERNS:
            if pattern.search(text):
                max_score = max(max_score, score)
                reasons.append(reason)

        passed = max_score < threshold
        if not passed:
            return self._violation(ctx, max_score, threshold, f"Regex matches: {', '.join(reasons)}", {"matched_bias_rules": reasons})

        # 2. Tier-2 Bias Projections (Sentence-Transformers)
        if getattr(config, "bias_projection_enabled", False):
            # Run projection synchronously in executor if needed, but it's fast enough on CPU
            proj_skew, proj_reason = self._projection_scorer.score(text)
            if proj_skew >= threshold:
                return self._violation(ctx, proj_skew, threshold, proj_reason, {"skewed_axis": proj_reason})

        # 3. Tier-3 Counterfactual Checker (Ollama SLM)
        if getattr(config, "counterfactual_check_enabled", False):
            ollama_url = getattr(config, "counterfactual_ollama_base_url", "http://localhost:11434/api/generate")
            model_name = getattr(config, "counterfactual_ollama_model_name", "batiai/gemma4-e2b:q4")
            api_key = ctx.metadata.get("openrouter_api_key")
            
            is_biased, cf_reason = await self._counterfactual_scorer.score(text, ollama_url, model_name, api_key)
            if is_biased:
                # Assign maximum severity score if SLM judges it biased
                return self._violation(ctx, 1.0, threshold, cf_reason, {"counterfactual": True})

        return self._allow()

    def _allow(self) -> DetectionResult:
        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            suggested_action=PDPAction.ALLOW,
        )

    def _violation(self, ctx: GuardContext, score: float, threshold: float, reason_detail: str, metadata: dict) -> DetectionResult:
        action = (
            PDPAction.REWRITE
            if ctx.policy.post_execution.action_on_violation == "rewrite"
            else PDPAction.BLOCK
        )
        reason_str = f"Bias detected (score {score:.2f} >= threshold {threshold:.2f}): {reason_detail}"
        logger.warning("Trace %s: %s", ctx.trace_id, reason_str)
        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=False,
            score=score,
            suggested_action=action,
            violation_code=ViolationCode.BIAS,
            reason=reason_str,
            metadata=metadata,
        )
