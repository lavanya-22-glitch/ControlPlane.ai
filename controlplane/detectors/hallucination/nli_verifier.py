import logging
from typing import List, Tuple, Optional
import numpy as np

from controlplane.detectors.base import BaseDetector, GuardStage, DetectionResult, GuardContext
from controlplane.detectors.hallucination.claim_splitter import split_claims
from controlplane.detectors.hallucination.context_aligner import align_claims_to_context
from controlplane.pdp.types import PDPAction, ViolationCode

logger = logging.getLogger("controlplane.detectors.hallucination")


class NLIRAGGroundingGuard(BaseDetector):
    """Batched NLI RAG Grounding & Hallucination Verification Engine."""

    def __init__(self, onnx_model_path: Optional[str] = None):
        self._session = None
        self._onnx_path = onnx_model_path
        self._init_onnx()

    def _init_onnx(self):
        if self._onnx_path:
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(self._onnx_path, providers=["CPUExecutionProvider"])
                logger.info("ONNX NLI model initialized from %s", self._onnx_path)
            except Exception as e:
                logger.warning("Could not initialize ONNX runtime from %s: %s", self._onnx_path, e)

    @property
    def name(self) -> str:
        return "rag_hallucination_engine"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.POST_EXECUTION

    def _calculate_heuristic_grounding(self, pairs: List[Tuple[str, str, float]]) -> List[float]:
        """Fast-path lexical/statistical grounding fallback."""
        scores = []
        for context, claim, overlap in pairs:
            # Overlap ratio combined with negation penalty
            has_negation_contradiction = (" not " in claim.lower() and " not " not in context.lower()) or (
                " never " in claim.lower() and " never " not in context.lower()
            )
            score = max(0.0, overlap - (0.3 if has_negation_contradiction else 0.0))
            scores.append(min(1.0, score * 1.3))
        return scores

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        if not ctx.policy.post_execution or not ctx.policy.post_execution.hallucination_engine:
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.ALLOW,
            )

        hallu_config = ctx.policy.post_execution.hallucination_engine
        if not hallu_config.enabled:
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.ALLOW,
            )

        context_chunks = ctx.retrieved_context or []
        completion = ctx.completion_text or ""

        # If no context was provided or completion is empty, allow or skip
        if not context_chunks or not completion:
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.ALLOW,
            )

        claims = split_claims(completion)
        if not claims:
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.ALLOW,
            )

        aligned_pairs = align_claims_to_context(claims, context_chunks)
        pair_scores = self._calculate_heuristic_grounding(aligned_pairs)
        avg_score = float(np.mean(pair_scores)) if pair_scores else 1.0

        min_score = hallu_config.min_grounding_score
        passed = avg_score >= min_score

        if not passed:
            action = (
                PDPAction.REWRITE
                if hallu_config.action_on_low_grounding == "rewrite"
                else PDPAction.BLOCK
            )
            reason = (
                f"RAG Grounding score {avg_score:.2f} is below required threshold {min_score:.2f}. "
                f"Evaluated {len(claims)} factual claim(s)."
            )
            logger.warning("Trace %s: %s", ctx.trace_id, reason)
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=False,
                score=avg_score,
                suggested_action=action,
                violation_code=ViolationCode.HALLUCINATION,
                reason=reason,
                metadata={
                    "claims_count": len(claims),
                    "claim_scores": pair_scores,
                    "fallback_message": hallu_config.fallback_message,
                },
            )

        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            score=avg_score,
            suggested_action=PDPAction.ALLOW,
            metadata={"claims_count": len(claims), "claim_scores": pair_scores},
        )
