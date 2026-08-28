"""
Agent Confidence Mismatch Detector.

Problem:
  An LLM can be "certain-sounding" in natural language but internally
  uncertain about the tool arguments it emits. When the model has low
  per-token confidence on the tool-call JSON, allowing the call risks
  executing an action the model itself wasn't sure about.

What this detector does:
  1. Extracts token-level log-probabilities (logprobs) from the upstream
     model response for the tool-call argument tokens.
  2. Computes:
       avg_logprob        — mean log-prob across argument tokens
       min_logprob        — lowest single-token confidence
       entropy            — uncertainty spread across top alternatives
       pct_low_confidence — fraction of tokens below per_token_threshold
  3. Flags a CONFIDENCE_MISMATCH violation if:
       - avg_logprob < avg_logprob_threshold  (overall low confidence)
       - OR pct_low_confidence > max_low_conf_pct  (too many uncertain tokens)
       - OR min_logprob < min_logprob_threshold     (single token disaster)

Model API assumptions (OpenAI-compatible):
  response["choices"][0]["logprobs"]["content"] = [
      {"token": "...", "logprob": -0.23, "top_logprobs": [...]},
      ...
  ]
  The detector is given the raw upstream JSON response in ctx.metadata["raw_response"].

Latency: <1ms — pure numpy arithmetic, no model calls.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from controlplane.detectors.base import BaseDetector, DetectionResult, GuardContext, GuardStage
from controlplane.pdp.types import PDPAction, ViolationCode

logger = logging.getLogger("controlplane.detectors.tool_safety.confidence")


# ---------------------------------------------------------------------------
# Logprob extraction utilities
# ---------------------------------------------------------------------------

def _extract_logprob_tokens(raw_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull the logprobs token list from an OpenAI-compatible response dict.

    Returns the flat list of token logprob objects, or [] if not available.
    Handles both chat completions (logprobs.content) and legacy completions.
    """
    try:
        choice = (raw_response.get("choices") or [{}])[0]
        logprobs_obj = choice.get("logprobs") or {}
        # OpenAI chat: {"content": [{token, logprob, top_logprobs},...]}
        content_tokens = logprobs_obj.get("content")
        if content_tokens and isinstance(content_tokens, list):
            return content_tokens
        # Legacy completions format: {"tokens": [...], "token_logprobs": [...]}
        tokens = logprobs_obj.get("tokens", [])
        token_logprobs = logprobs_obj.get("token_logprobs", [])
        if tokens and token_logprobs:
            return [
                {"token": t, "logprob": lp}
                for t, lp in zip(tokens, token_logprobs)
                if lp is not None
            ]
    except Exception:
        pass
    return []


def _token_entropy(top_logprobs: List[Dict[str, Any]]) -> float:
    """Shannon entropy from the top-k alternative log-probs for a single token.

    H = -Σ p_i * log(p_i)
    Higher entropy = model was more uncertain about this token.
    """
    if not top_logprobs:
        return 0.0
    lps = np.array([item.get("logprob", -20.0) for item in top_logprobs], dtype=float)
    probs = np.exp(np.clip(lps, -20.0, 0.0))
    probs = probs / (probs.sum() + 1e-12)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return entropy


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ConfidenceMetrics:
    """Computed confidence metrics for a single tool call response."""
    avg_logprob: float          # Mean log-prob across all argument tokens
    min_logprob: float          # Worst single-token log-prob
    avg_entropy: float          # Mean per-token entropy (top-k alternatives)
    pct_low_confidence: float   # Fraction of tokens below per_token_threshold
    token_count: int            # How many tokens were scored

    @property
    def avg_probability(self) -> float:
        """Average per-token probability (geometric mean via exp(avg_logprob))."""
        return math.exp(self.avg_logprob)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_logprob": round(self.avg_logprob, 4),
            "min_logprob": round(self.min_logprob, 4),
            "avg_entropy": round(self.avg_entropy, 4),
            "pct_low_confidence": round(self.pct_low_confidence, 3),
            "avg_probability": round(self.avg_probability, 4),
            "token_count": self.token_count,
        }


def _compute_confidence_metrics(
    token_list: List[Dict[str, Any]],
    per_token_threshold: float,
) -> ConfidenceMetrics:
    """Compute aggregate confidence metrics from a token logprob list."""
    if not token_list:
        # No logprobs available — return neutral/unknown metrics
        return ConfidenceMetrics(
            avg_logprob=0.0,
            min_logprob=0.0,
            avg_entropy=0.0,
            pct_low_confidence=0.0,
            token_count=0,
        )

    logprobs = np.array(
        [t.get("logprob", 0.0) for t in token_list if t.get("logprob") is not None],
        dtype=float,
    )
    entropies = np.array(
        [_token_entropy(t.get("top_logprobs", [])) for t in token_list],
        dtype=float,
    )

    if len(logprobs) == 0:
        return ConfidenceMetrics(0.0, 0.0, 0.0, 0.0, 0)

    low_conf_count = int(np.sum(logprobs < per_token_threshold))

    return ConfidenceMetrics(
        avg_logprob=float(np.mean(logprobs)),
        min_logprob=float(np.min(logprobs)),
        avg_entropy=float(np.mean(entropies)),
        pct_low_confidence=low_conf_count / len(logprobs),
        token_count=len(logprobs),
    )


# ---------------------------------------------------------------------------
# ConfidenceMismatchGuard
# ---------------------------------------------------------------------------

class ConfidenceMismatchGuard(BaseDetector):
    """Agent tool-call confidence mismatch detector.

    Reads token-level logprobs from the upstream model response and raises
    a CONFIDENCE_MISMATCH flag when the model's internal certainty about
    the tool-call arguments falls below configured thresholds.

    Requires: ctx.metadata["raw_response"] to contain the full upstream
              JSON response with logprobs enabled.

    When logprobs are absent (model does not support them, or the upstream
    did not request logprob=True), the guard passes with a warning in metadata.
    """

    @property
    def name(self) -> str:
        return "confidence_mismatch_guard"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.TOOL_INTERCEPTION

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        agent_config = _get_agent_config(ctx)
        conf_cfg = getattr(agent_config, "confidence_mismatch", None)

        if agent_config is None or conf_cfg is None or not getattr(conf_cfg, "enabled", True):
            return self._allow(metadata={"reason": "confidence_check_disabled"})

        # No tool calls — nothing to score
        if not ctx.tool_calls:
            return self._allow(metadata={"reason": "no_tool_calls"})

        # Retrieve the raw upstream response stored in context metadata
        raw_response: Dict[str, Any] = ctx.metadata.get("raw_response", {})
        token_list = _extract_logprob_tokens(raw_response)

        if not token_list:
            logger.info(
                "Trace %s: ConfidenceMismatchGuard: no logprobs in response. "
                "Enable logprobs=True in upstream request for confidence scoring.",
                ctx.trace_id,
            )
            return self._allow(
                metadata={
                    "reason": "logprobs_unavailable",
                    "suggestion": "Set logprobs=True in upstream request to enable confidence scoring.",
                }
            )

        # Config thresholds (with defaults)
        avg_threshold: float = getattr(conf_cfg, "avg_logprob_threshold", -2.0)
        min_threshold: float = getattr(conf_cfg, "min_logprob_threshold", -4.0)
        per_token_threshold: float = getattr(conf_cfg, "per_token_threshold", -2.5)
        max_low_conf_pct: float = getattr(conf_cfg, "max_low_conf_pct", 0.30)

        metrics = _compute_confidence_metrics(token_list, per_token_threshold)

        violations: List[str] = []

        if metrics.avg_logprob < avg_threshold:
            violations.append(
                f"avg_logprob {metrics.avg_logprob:.3f} < threshold {avg_threshold:.2f} "
                f"(model avg certainty too low)"
            )

        if metrics.min_logprob < min_threshold:
            violations.append(
                f"min_logprob {metrics.min_logprob:.3f} < threshold {min_threshold:.2f} "
                f"(single token catastrophically uncertain)"
            )

        if metrics.pct_low_confidence > max_low_conf_pct:
            violations.append(
                f"{metrics.pct_low_confidence:.0%} tokens below per_token_threshold "
                f"{per_token_threshold:.2f} (exceeds max {max_low_conf_pct:.0%})"
            )

        metadata = {
            "tool_calls": [
                tc.get("function", {}).get("name") for tc in (ctx.tool_calls or [])
            ],
            "confidence_metrics": metrics.to_dict(),
            "thresholds": {
                "avg_logprob_threshold": avg_threshold,
                "min_logprob_threshold": min_threshold,
                "per_token_threshold": per_token_threshold,
                "max_low_conf_pct": max_low_conf_pct,
            },
            "violations_detail": violations,
        }

        if violations:
            reason = (
                f"Agent confidence mismatch detected on tool call(s) "
                f"{[tc.get('function', {}).get('name') for tc in ctx.tool_calls]}: "
                + "; ".join(violations)
            )
            logger.warning("Trace %s: %s", ctx.trace_id, reason)
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=False,
                score=metrics.avg_probability,
                suggested_action=PDPAction.BLOCK,
                violation_code=ViolationCode.AGENT_CONFIDENCE_MISMATCH,
                reason=reason,
                metadata=metadata,
            )

        logger.debug(
            "Trace %s: Confidence check PASSED. avg_logprob=%.3f, min=%.3f, pct_low=%.1f%%",
            ctx.trace_id,
            metrics.avg_logprob,
            metrics.min_logprob,
            metrics.pct_low_confidence * 100,
        )
        return self._allow(score=metrics.avg_probability, metadata=metadata)

    def _allow(
        self, score: Optional[float] = None, metadata: Optional[Dict] = None
    ) -> DetectionResult:
        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            score=score,
            suggested_action=PDPAction.ALLOW,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_agent_config(ctx: GuardContext):
    """Retrieve the agent-specific config from the policy, or None."""
    # Agent config lives at policy.agent_config (to be added to models.py)
    return getattr(ctx.policy, "agent_config", None)
