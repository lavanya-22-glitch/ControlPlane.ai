"""
Reasoning & Output Similarity Checker for Agent Tool Calls.

Problem:
  When a model "thinks out loud" (chain-of-thought reasoning tokens), its
  reasoning should be consistent with the tool arguments it finally emits.
  A large semantic gap between reasoning and output is a signal of:
    - Reasoning hijacking (malicious context steering the reasoning)
    - Mode collapse / copy-paste hallucination in tool arguments
    - Prompt injection redirecting the thinking but not the output (or vice-versa)

How reasoning tokens are exposed (assumption: model thinking is revealed):
  Option A — Separate field (Anthropic extended thinking, o-series):
    response["choices"][0]["message"]["reasoning_content"] = "<thinking>...</thinking>"

  Option B — Inline tags in the completion:
    completion_text = "<thinking>I should call get_user...</thinking>\ncall_tool(...)"

  Option C — Passed explicitly in ctx.metadata["reasoning_text"] by the caller
    (most portable — works regardless of model provider format).

The detector handles all three cases, in priority order: C > A > B.

Similarity scoring (two tiers, same pattern as citation_similarity.py):
  Tier 1 — N-gram overlap (zero dependencies, <0.5ms):
    Bigram Dice + content-word recall.

  Tier 2 — Sentence embedding cosine (optional, sentence-transformers):
    Bi-encoder (MiniLM-L6-v2) encoding reasoning and tool-arg text.
    More accurate for paraphrasing / semantic mismatch.

Short-circuit: a single similarity score below `similarity_fail_threshold`
immediately returns FAIL — no further processing.

Latency budget:
  N-gram (always): <1ms per check
  Embedding (optional): ~10-30ms per check (MiniLM-L6, CPU)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from controlplane.detectors.base import BaseDetector, DetectionResult, GuardContext, GuardStage
from controlplane.pdp.types import PDPAction, ViolationCode

logger = logging.getLogger("controlplane.detectors.tool_safety.reasoning_similarity")

_REASONING_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reasoning_sim")

# Inline thinking tag patterns (case-insensitive)
_THINKING_TAG_RE = re.compile(
    r'<(?:thinking|reasoning|scratchpad)>(.*?)</(?:thinking|reasoning|scratchpad)>',
    re.DOTALL | re.IGNORECASE,
)

# Stopwords for content-word filtering
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "and", "or", "but", "if", "as",
    "this", "that", "it", "its", "their", "our", "we", "they", "he",
    "she", "i", "you", "from", "into", "not", "no", "any", "all", "so",
    "also", "then", "just", "more", "want", "need", "use", "make",
    "call", "function", "tool", "true", "false", "null", "none",
})


# ---------------------------------------------------------------------------
# Reasoning extraction
# ---------------------------------------------------------------------------

def extract_reasoning_text(ctx: GuardContext) -> Optional[str]:
    """Extract chain-of-thought / reasoning tokens from the guard context.

    Priority:
      1. ctx.metadata["reasoning_text"]             — caller-provided (best)
      2. raw_response message.reasoning_content      — Anthropic / o-series extended thinking
      3. <thinking>...</thinking> tags in completion — inline CoT tags
    """
    # Priority 1: explicit caller-provided reasoning text
    reasoning = ctx.metadata.get("reasoning_text")
    if reasoning and reasoning.strip():
        logger.debug("Reasoning extracted from ctx.metadata['reasoning_text']")
        return reasoning.strip()

    # Priority 2: reasoning_content field in raw response message
    raw_response: Dict[str, Any] = ctx.metadata.get("raw_response", {})
    try:
        message = (raw_response.get("choices") or [{}])[0].get("message", {})
        reasoning_content = message.get("reasoning_content") or message.get("thinking")
        if reasoning_content and isinstance(reasoning_content, str) and reasoning_content.strip():
            logger.debug("Reasoning extracted from response.message.reasoning_content")
            return reasoning_content.strip()
    except Exception:
        pass

    # Priority 3: inline <thinking>...</thinking> tags in completion text
    completion = ctx.completion_text or ""
    matches = _THINKING_TAG_RE.findall(completion)
    if matches:
        reasoning = " ".join(m.strip() for m in matches)
        logger.debug("Reasoning extracted from inline <thinking> tags in completion")
        return reasoning.strip()

    return None


def extract_tool_arg_text(tool_calls: List[Dict[str, Any]]) -> str:
    """Serialise tool-call arguments to a single text blob for similarity comparison.

    Produces human-readable key=value pairs rather than raw JSON so that
    similarity scoring is not confused by JSON punctuation tokens.
    """
    parts: List[str] = []
    for tc in tool_calls or []:
        func = tc.get("function", {})
        name = func.get("name", "unknown_tool")
        raw_args = func.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            if isinstance(args, dict):
                # "tool_name: key=value key2=value2"
                kv = " ".join(f"{k}={v}" for k, v in args.items())
                parts.append(f"{name}: {kv}")
            else:
                parts.append(f"{name}: {args}")
        except Exception:
            parts.append(f"{name}: {raw_args}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Similarity scorers
# ---------------------------------------------------------------------------

class ReasoningSimScorer(ABC):
    """Abstract similarity scorer for (reasoning_text, tool_arg_text) pairs."""

    @property
    @abstractmethod
    def scorer_name(self) -> str: ...

    @abstractmethod
    def score(self, reasoning: str, tool_args: str) -> float:
        """Return similarity in [0, 1]. Higher = more aligned."""
        ...


class NgramReasoningScorer(ReasoningSimScorer):
    """Bigram Dice coefficient + content-word recall.

    Combined score = 0.5 * bigram_dice + 0.5 * content_word_jaccard

    Rationale for equal weighting (vs. citation scorer's 0.6/0.4):
      Reasoning text is longer and richer than citation preceding-text;
      Jaccard between content words is a better proxy for topic alignment.
    """

    @property
    def scorer_name(self) -> str:
        return "ngram"

    @staticmethod
    def _bigrams(text: str) -> set:
        t = text.lower()
        return {t[i:i+2] for i in range(len(t) - 1)} if len(t) >= 2 else set()

    @staticmethod
    def _content_words(text: str) -> frozenset:
        words = re.findall(r"\b\w{3,}\b", text.lower())
        return frozenset(w for w in words if w not in _STOPWORDS)

    def score(self, reasoning: str, tool_args: str) -> float:
        if not reasoning or not tool_args:
            return 0.0

        bg_r = self._bigrams(reasoning)
        bg_t = self._bigrams(tool_args)
        dice = (2 * len(bg_r & bg_t) / (len(bg_r) + len(bg_t))) if (bg_r or bg_t) else 0.0

        cw_r = self._content_words(reasoning)
        cw_t = self._content_words(tool_args)
        union = cw_r | cw_t
        jaccard = (len(cw_r & cw_t) / len(union)) if union else 0.0

        return 0.5 * dice + 0.5 * jaccard


class EmbeddingReasoningScorer(ReasoningSimScorer):
    """Sentence embedding cosine similarity (bi-encoder).

    More accurate than n-gram for paraphrasing and semantic reordering.
    Requires sentence-transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._model.encode(["warmup"], show_progress_bar=False)
            logger.info("EmbeddingReasoningScorer loaded: %s", self._model_name)
        except Exception as exc:
            logger.error("EmbeddingReasoningScorer load failed: %s", exc)
            raise

    @property
    def scorer_name(self) -> str:
        return f"embedding:{self._model_name}"

    def score(self, reasoning: str, tool_args: str) -> float:
        if not reasoning or not tool_args or self._model is None:
            return 0.0
        embs = self._model.encode(
            [reasoning[:512], tool_args[:256]],  # cap length to keep latency bounded
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return float(np.dot(embs[0], embs[1]))


def _build_reasoning_scorer(method: str = "auto") -> ReasoningSimScorer:
    """Factory: auto-select best available scorer."""
    if method == "ngram":
        return NgramReasoningScorer()
    if method == "embedding":
        return EmbeddingReasoningScorer()
    # auto: try embedding, fall back to ngram
    try:
        scorer = EmbeddingReasoningScorer()
        logger.info("Reasoning similarity scorer: embedding")
        return scorer
    except Exception as exc:
        logger.info("Embedding scorer unavailable (%s); using n-gram.", exc)
        return NgramReasoningScorer()


# ---------------------------------------------------------------------------
# ReasoningOutputSimilarityGuard
# ---------------------------------------------------------------------------

class ReasoningOutputSimilarityGuard(BaseDetector):
    """Detects semantic misalignment between agent reasoning and tool-call output.

    When the model's chain-of-thought reasoning discusses one action but the
    emitted tool-call arguments reflect a different action, this is a strong
    signal of reasoning hijacking, prompt injection, or confabulation.

    Requires:
      ctx.tool_calls              — list of tool calls from the response
      ctx.completion_text OR
      ctx.metadata["reasoning_text"] OR
      ctx.metadata["raw_response"].choices[0].message.reasoning_content

    When no reasoning text is available, the guard passes with a
    'reasoning_unavailable' note in metadata.
    """

    _scorer_singleton: Optional[ReasoningSimScorer] = None

    def __init__(self, scorer: Optional[ReasoningSimScorer] = None) -> None:
        self._scorer = scorer

    def _get_scorer(self, method: str = "auto") -> ReasoningSimScorer:
        if self._scorer is not None:
            return self._scorer
        if ReasoningOutputSimilarityGuard._scorer_singleton is None:
            ReasoningOutputSimilarityGuard._scorer_singleton = _build_reasoning_scorer(method)
        return ReasoningOutputSimilarityGuard._scorer_singleton

    @property
    def name(self) -> str:
        return "reasoning_output_similarity_guard"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.TOOL_INTERCEPTION

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        from controlplane.detectors.tool_safety.confidence_mismatch import _get_agent_config

        agent_config = _get_agent_config(ctx)
        ros_cfg = getattr(agent_config, "reasoning_similarity", None)

        if agent_config is None or ros_cfg is None or not getattr(ros_cfg, "enabled", True):
            return self._allow(metadata={"reason": "reasoning_similarity_check_disabled"})

        if not ctx.tool_calls:
            return self._allow(metadata={"reason": "no_tool_calls"})

        # ---- Step 1: Extract reasoning text --------------------------------
        reasoning_text = extract_reasoning_text(ctx)
        if not reasoning_text:
            logger.info(
                "Trace %s: No reasoning text found. "
                "Provide reasoning via ctx.metadata['reasoning_text'], "
                "response.message.reasoning_content, or <thinking> tags.",
                ctx.trace_id,
            )
            return self._allow(
                metadata={
                    "reason": "reasoning_unavailable",
                    "suggestion": (
                        "Expose chain-of-thought via ctx.metadata['reasoning_text'], "
                        "response.message.reasoning_content, or <thinking>…</thinking> tags."
                    ),
                }
            )

        # ---- Step 2: Serialise tool args to text ---------------------------
        tool_arg_text = extract_tool_arg_text(ctx.tool_calls)
        if not tool_arg_text.strip():
            return self._allow(metadata={"reason": "empty_tool_args"})

        # ---- Step 3: Compute similarity (offloaded to thread pool) ---------
        method: str = getattr(ros_cfg, "similarity_method", "auto")
        fail_threshold: float = getattr(ros_cfg, "similarity_fail_threshold", 0.15)

        loop = asyncio.get_event_loop()
        similarity_score: float = await loop.run_in_executor(
            _REASONING_EXECUTOR,
            self._compute_similarity,
            reasoning_text,
            tool_arg_text,
            method,
        )

        tool_names = [tc.get("function", {}).get("name") for tc in ctx.tool_calls]
        metadata = {
            "reasoning_preview": reasoning_text[:200],
            "tool_arg_preview": tool_arg_text[:200],
            "tool_names": tool_names,
            "similarity_score": round(similarity_score, 4),
            "fail_threshold": fail_threshold,
            "scorer": self._get_scorer(method).scorer_name,
        }

        if similarity_score < fail_threshold:
            reason = (
                f"Reasoning-output similarity {similarity_score:.3f} < threshold "
                f"{fail_threshold:.2f} for tool(s) {tool_names}. "
                "Model reasoning does not align with the tool arguments emitted — "
                "possible reasoning hijacking or confabulation."
            )
            logger.warning("Trace %s: %s", ctx.trace_id, reason)
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=False,
                score=similarity_score,
                suggested_action=PDPAction.BLOCK,
                violation_code=ViolationCode.AGENT_REASONING_MISMATCH,
                reason=reason,
                metadata=metadata,
            )

        logger.debug(
            "Trace %s: Reasoning similarity PASSED. score=%.3f threshold=%.2f tool=%s",
            ctx.trace_id,
            similarity_score,
            fail_threshold,
            tool_names,
        )
        return self._allow(score=similarity_score, metadata=metadata)

    def _compute_similarity(self, reasoning: str, tool_args: str, method: str) -> float:
        return self._get_scorer(method).score(reasoning, tool_args)

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
