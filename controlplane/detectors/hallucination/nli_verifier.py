"""
NLI RAG Grounding & Hallucination Verification Engine.

Full Pipeline (per request):
    1. split_claims()              — sentence-level factual claim extraction
    2. context_reranker (optional) — filter irrelevant RAG chunks via cross-encoder reranker
    3. align_claims_to_context()   — match each claim → best RAG chunk (lexical, O(N×C))
    4. Lexical pre-filter gate     — skip NLI for obvious HIGH-confidence passes
    5. async_score_pairs()         — batched NLI inference (single forward pass)
    6. Per-claim early exit        — abort on first claim below early_exit_threshold
    7. Aggregate avg grounding     — compare to min_grounding_score → PDP action

Latency controls:
    - Context pre-tokenization is done once, not per-claim.
    - Claims scoring ≥ lexical_pass_gate skip NLI inference entirely.
    - Claims scoring < early_exit_threshold cause immediate FAIL (no remaining claims scored).
    - NLI inference is offloaded to a thread-pool executor (never blocks the event loop).
    - Model/tokenizer is a module-level singleton (loaded once at startup).
    - Context reranker filters irrelevant chunks before NLI, shrinking the scoring problem.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from controlplane.detectors.base import BaseDetector, DetectionResult, GuardContext, GuardStage
from controlplane.detectors.hallucination.claim_splitter import split_claims
from controlplane.detectors.hallucination.context_aligner import align_claims_to_context
from controlplane.detectors.hallucination.context_reranker import get_context_reranker
from controlplane.detectors.hallucination.nli_backend import (
    HeuristicNLIBackend,
    NLIBackend,
    NLIResult,
    auto_select_backend,
)
from controlplane.detectors.hallucination.types import ClaimContextPair
from controlplane.pdp.types import PDPAction, ViolationCode

logger = logging.getLogger("controlplane.detectors.hallucination")

# Lexical overlap threshold above which we skip NLI inference (obvious pass).
# Claims that share ≥80% tokens with their context chunk are trivially grounded.
_LEXICAL_PASS_GATE: float = 0.80


class NLIRAGGroundingGuard(BaseDetector):
    """Batched NLI RAG Grounding & Hallucination Verification Engine.

    Args:
        backend: Explicit ``NLIBackend`` instance. If ``None``, the auto-selected
                 singleton backend is used (recommended for production — model is
                 loaded once and shared across all requests).

    Usage::
        guard = NLIRAGGroundingGuard()          # auto backend, singleton warm-up
        guard = NLIRAGGroundingGuard(backend=HeuristicNLIBackend())  # for tests
    """

    def __init__(self, backend: Optional[NLIBackend] = None) -> None:
        # Use caller-provided backend (useful for tests) or the auto singleton.
        # auto_select_backend() caches its result so the model is never reloaded.
        self._backend: NLIBackend = backend or auto_select_backend()

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "rag_hallucination_engine"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.POST_EXECUTION

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        hallu_config = (
            ctx.policy.post_execution.hallucination_engine
            if ctx.policy.post_execution
            else None
        )

        # Short-circuit: guard not configured or disabled
        if hallu_config is None or not hallu_config.enabled:
            return self._allow(score=None, metadata={})

        context_chunks: List[str] = ctx.retrieved_context or []
        completion: str = ctx.completion_text or ""

        # Short-circuit: nothing to evaluate
        if not context_chunks or not completion:
            return self._allow(score=None, metadata={"reason": "no_context_or_completion"})

        # ---- Step 1: Claim extraction ----------------------------------------
        t0 = time.perf_counter()
        claims = split_claims(completion)
        if not claims:
            return self._allow(score=None, metadata={"reason": "no_claims_extracted"})

        # ---- Step 2: Context reranking (optional) ----------------------------
        #   Re-scores retrieved chunks against the user query using a cross-encoder
        #   reranker. Filters out off-topic chunks that could cause NLI false positives.
        reranker_metadata: dict = {"enabled": False}
        reranking_enabled: bool = getattr(hallu_config, "context_reranking_enabled", False)

        if reranking_enabled:
            # Extract the original user query from the last user message
            user_query = ""
            for msg in reversed(ctx.messages):
                if msg.get("role") == "user":
                    user_query = msg.get("content", "")
                    break

            if user_query:
                reranker_model = getattr(
                    hallu_config, "context_reranker_model",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                reranker_threshold = getattr(
                    hallu_config, "context_relevance_threshold", 0.30
                )
                reranker_top_k = getattr(hallu_config, "context_reranker_top_k", None)

                reranker = get_context_reranker(
                    model_name=reranker_model,
                    relevance_threshold=reranker_threshold,
                    top_k=reranker_top_k,
                )

                if reranker is not None:
                    t_rerank_start = time.perf_counter()
                    ranked_chunks = await reranker.async_rerank(
                        query=user_query,
                        chunks=context_chunks,
                    )
                    t_rerank_ms = (time.perf_counter() - t_rerank_start) * 1000

                    # Replace context_chunks with only the relevant ones
                    context_chunks = [r.text for r in ranked_chunks]
                    reranker_metadata = {
                        "enabled": True,
                        "chunks_before": len(ctx.retrieved_context or []),
                        "chunks_after": len(context_chunks),
                        "reranker_model": reranker_model,
                        "latency_ms": round(t_rerank_ms, 1),
                        "top_scores": [
                            round(r.relevance_score, 3) for r in ranked_chunks[:5]
                        ],
                    }
                    logger.debug(
                        "Trace %s: Reranker filtered %d → %d chunks [%.1fms]",
                        ctx.trace_id,
                        reranker_metadata["chunks_before"],
                        reranker_metadata["chunks_after"],
                        t_rerank_ms,
                    )

        # ---- Step 3: Lexical alignment -----------------------------------------
        aligned: List[ClaimContextPair] = align_claims_to_context(claims, context_chunks)

        # ---- Step 3: Lexical pre-filter — skip NLI for obvious passes ----------
        #   Claims with lexical_overlap >= _LEXICAL_PASS_GATE are trivially grounded;
        #   assign synthetic NLIResult(entailment=1.0) and skip inference.
        needs_nli: List[ClaimContextPair] = []
        prefilter_results: dict[int, NLIResult] = {}

        for idx, pair in enumerate(aligned):
            if pair.lexical_overlap >= _LEXICAL_PASS_GATE:
                # Clearly grounded — no need to burn inference budget
                prefilter_results[idx] = NLIResult(
                    entailment=1.0, contradiction=0.0, neutral=0.0
                )
            else:
                needs_nli.append(pair)

        t_prefilter = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Trace %s: %d/%d claims sent to NLI (pre-filter skipped %d) [%.1fms]",
            ctx.trace_id,
            len(needs_nli),
            len(claims),
            len(prefilter_results),
            t_prefilter,
        )

        # ---- Step 5: Batched NLI inference (single async call) -----------------
        nli_results_list: List[NLIResult] = []
        if needs_nli:
            t_nli_start = time.perf_counter()
            pairs_for_nli = [(p.context_chunk, p.claim) for p in needs_nli]
            nli_results_list = await self._backend.async_score_pairs(pairs_for_nli)
            t_nli_ms = (time.perf_counter() - t_nli_start) * 1000
            logger.debug(
                "Trace %s: NLI inference for %d pairs completed in %.1fms (backend=%s)",
                ctx.trace_id,
                len(needs_nli),
                t_nli_ms,
                self._backend.backend_name,
            )

        # Merge pre-filter results back into full ordered list
        nli_iter = iter(nli_results_list)
        all_results: List[NLIResult] = []
        for idx in range(len(aligned)):
            if idx in prefilter_results:
                all_results.append(prefilter_results[idx])
            else:
                all_results.append(next(nli_iter))

        # ---- Step 6: Per-claim short-circuit (early exit on first bad claim) ---
        early_exit_threshold: float = getattr(hallu_config, "early_exit_threshold", 0.20)
        per_claim_metadata = []

        for pair, result in zip(aligned, all_results):
            entry = {
                "claim": pair.claim,
                "context_chunk": pair.context_chunk[:120],  # truncate for audit compactness
                "lexical_overlap": round(pair.lexical_overlap, 3),
                "entailment": round(result.entailment, 3),
                "contradiction": round(result.contradiction, 3),
                "neutral": round(result.neutral, 3),
                "grounding_score": round(result.grounding_score, 3),
                "via_prefilter": pair.lexical_overlap >= _LEXICAL_PASS_GATE,
            }
            per_claim_metadata.append(entry)

            # Short-circuit: this claim is dangerously ungrounded
            if result.grounding_score < early_exit_threshold:
                reason = (
                    f"Early exit: claim grounding score {result.grounding_score:.3f} "
                    f"< early_exit_threshold {early_exit_threshold:.2f}. "
                    f"Claim: \"{pair.claim[:80]}...\""
                )
                logger.warning("Trace %s: %s", ctx.trace_id, reason)
                return self._fail(
                    score=result.grounding_score,
                    hallu_config=hallu_config,
                    reason=reason,
                    metadata={
                        "claims_evaluated": len(per_claim_metadata),
                        "claims_total": len(claims),
                        "early_exit": True,
                        "per_claim_scores": per_claim_metadata,
                        "backend": self._backend.backend_name,
                        "fallback_message": hallu_config.fallback_message,
                    },
                )

        # ---- Step 6: Aggregate grounding score ---------------------------------
        grounding_scores = [r.grounding_score for r in all_results]
        avg_score = float(np.mean(grounding_scores)) if grounding_scores else 1.0
        min_score: float = hallu_config.min_grounding_score
        passed = avg_score >= min_score

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Trace %s: RAG grounding avg=%.3f min=%.2f passed=%s [%.1fms total, backend=%s]",
            ctx.trace_id,
            avg_score,
            min_score,
            passed,
            total_ms,
            self._backend.backend_name,
        )

        metadata = {
            "claims_total": len(claims),
            "claims_evaluated": len(all_results),
            "avg_grounding_score": round(avg_score, 3),
            "per_claim_scores": per_claim_metadata,
            "early_exit": False,
            "backend": self._backend.backend_name,
            "latency_ms": round(total_ms, 1),
            "context_reranker": reranker_metadata,
        }

        if not passed:
            reason = (
                f"RAG grounding score {avg_score:.3f} is below required threshold "
                f"{min_score:.2f} ({len(claims)} claim(s) evaluated)."
            )
            logger.warning("Trace %s: %s", ctx.trace_id, reason)
            metadata["fallback_message"] = hallu_config.fallback_message
            return self._fail(
                score=avg_score,
                hallu_config=hallu_config,
                reason=reason,
                metadata=metadata,
            )

        return self._allow(score=avg_score, metadata=metadata)

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _allow(self, score: Optional[float], metadata: dict) -> DetectionResult:
        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            score=score,
            suggested_action=PDPAction.ALLOW,
            metadata=metadata,
        )

    def _fail(self, score: float, hallu_config, reason: str, metadata: dict) -> DetectionResult:
        action = (
            PDPAction.REWRITE
            if hallu_config.action_on_low_grounding == "rewrite"
            else PDPAction.BLOCK
        )
        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=False,
            score=score,
            suggested_action=action,
            violation_code=ViolationCode.HALLUCINATION,
            reason=reason,
            metadata=metadata,
        )
