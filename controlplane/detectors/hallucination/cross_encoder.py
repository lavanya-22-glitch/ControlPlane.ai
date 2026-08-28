"""
Cross-Encoder NLI Backend — production implementation using sentence-transformers.

Why CrossEncoder instead of pipeline("text-classification"):
  - CrossEncoder performs *joint* encoding: the model attends over both
    (premise, hypothesis) simultaneously via full cross-attention layers.
  - This is fundamentally more accurate than bi-encoders (which encode each
    sequence independently and compare embeddings).
  - sentence_transformers.CrossEncoder gives us direct access to logits,
    proper batch handling, and activation function control.

Supported model families:
  NLI 3-class:  cross-encoder/nli-deberta-v3-small  (recommended, 184MB)
                cross-encoder/nli-MiniLM2-L6-H768   (smaller, faster)
                cross-encoder/nli-roberta-base       (robust)

Label schema detection:
  Different NLI models have different id→label orderings. This module
  reads the model's config.json at load time to resolve the exact mapping,
  rather than hard-coding assumptions.

Latency budget (CPU, fp32):
  cross-encoder/nli-MiniLM2-L6-H768   : ~15-40ms  per batch of 8 pairs
  cross-encoder/nli-deberta-v3-small   : ~40-120ms per batch of 8 pairs
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from controlplane.detectors.hallucination.nli_backend import NLIBackend, NLIResult

logger = logging.getLogger("controlplane.detectors.hallucination.cross_encoder")


# ---------------------------------------------------------------------------
# Label schema registry
# ---------------------------------------------------------------------------

# Fallback heuristic schemas when the model config cannot be loaded.
# Map: substring of model name → {label_name: output_index}
_KNOWN_SCHEMAS: Dict[str, Dict[str, int]] = {
    "deberta":  {"contradiction": 0, "entailment": 1, "neutral": 2},
    "roberta":  {"contradiction": 0, "entailment": 2, "neutral": 1},
    "minilm":   {"contradiction": 0, "entailment": 1, "neutral": 2},
    "albert":   {"contradiction": 0, "entailment": 1, "neutral": 2},
    "default":  {"contradiction": 0, "entailment": 1, "neutral": 2},
}


def _resolve_label_schema(model_name: str, id2label: Optional[Dict[int, str]]) -> Dict[str, int]:
    """Build a {label_name: output_index} mapping.

    Priority:
      1. Model config id2label (authoritative, loaded from HF hub).
      2. Name-based heuristic lookup (_KNOWN_SCHEMAS).
      3. Default fallback.
    """
    if id2label:
        schema = {label.lower(): idx for idx, label in id2label.items()}
        # Validate it contains the 3 expected labels
        if all(k in schema for k in ("entailment", "contradiction", "neutral")):
            logger.debug("Label schema resolved from model config: %s", schema)
            return schema
        logger.warning(
            "Model id2label missing expected keys, falling back to heuristic. Got: %s",
            id2label,
        )

    # Heuristic: match model name substring
    name_lower = model_name.lower()
    for key, schema in _KNOWN_SCHEMAS.items():
        if key != "default" and key in name_lower:
            logger.debug("Label schema resolved from heuristic key='%s': %s", key, schema)
            return schema

    logger.warning("Using default label schema for model '%s'", model_name)
    return _KNOWN_SCHEMAS["default"]


# ---------------------------------------------------------------------------
# CrossEncoderNLIBackend
# ---------------------------------------------------------------------------

class CrossEncoderNLIBackend(NLIBackend):
    """sentence-transformers CrossEncoder NLI backend.

    Encodes (context, claim) pairs jointly via cross-attention for accurate
    entailment scoring. More accurate than bi-encoders; more expensive than
    lexical heuristics.

    Args:
        model_name:  HuggingFace model identifier for a 3-class NLI cross-encoder.
        max_length:  Token truncation limit per sequence pair.
                     128 works well for sentence-level claims.
        batch_size:  Max pairs per forward pass. ≤16 recommended for CPU.
        device:      'cpu', 'cuda', 'cuda:0', etc.
                     None = auto-detect (CUDA if available, else CPU).

    Usage::
        backend = CrossEncoderNLIBackend()
        results = backend.score_pairs([
            ("Context: Refunds take 7 days.", "Refunds are processed in a week."),
            ("Context: No returns after 30 days.", "You can return items any time."),
        ])
        # results[0].grounding_score → high  (entailment)
        # results[1].grounding_score → low   (contradiction)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        max_length: int = 128,
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device
        self._model = None
        self._label_schema: Dict[str, int] = _KNOWN_SCHEMAS["default"]
        self._load()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load model, resolve label schema, and warm up."""
        try:
            from sentence_transformers import CrossEncoder
            import torch

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
                device=self._device,
            )

            # Resolve label schema from model config (authoritative)
            id2label = self._model.config.id2label if hasattr(self._model, "config") else None
            if id2label and isinstance(id2label, dict):
                # HF id2label uses int keys
                typed_id2label = {
                    int(k): v for k, v in id2label.items()
                    if str(k).isdigit() or isinstance(k, int)
                }
                self._label_schema = _resolve_label_schema(self._model_name, typed_id2label)
            else:
                self._label_schema = _resolve_label_schema(self._model_name, None)

            # Warm-up: single dummy inference so JIT / weights are hot
            self._model.predict(
                [("warmup context sentence here.", "warmup hypothesis claim here.")],
                batch_size=1,
                show_progress_bar=False,
                apply_softmax=True,
            )

            logger.info(
                "CrossEncoderNLIBackend ready: model=%s, device=%s, max_length=%d, schema=%s",
                self._model_name,
                self._device,
                self._max_length,
                self._label_schema,
            )

        except Exception as exc:
            logger.error("CrossEncoderNLIBackend failed to load '%s': %s", self._model_name, exc)
            raise

    # ------------------------------------------------------------------
    # NLIBackend interface
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return f"cross_encoder:{self._model_name}"

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """Score (context, claim) pairs using cross-encoder NLI.

        The model receives both the context chunk (premise) and the factual
        claim (hypothesis) concatenated with [SEP] and performs full cross-
        attention between them, producing per-class logits.

        Args:
            pairs: List of (context_chunk, claim) tuples.

        Returns:
            NLIResult per pair in the same order, with:
              - entailment:    P(claim is supported by context)
              - contradiction: P(claim contradicts context)
              - neutral:       P(claim is unrelated / indeterminate)
              - grounding_score: entailment − contradiction, clamped [0,1]
        """
        if not pairs or self._model is None:
            logger.warning("CrossEncoderNLIBackend: no pairs or model not loaded, returning defaults.")
            return [NLIResult(entailment=0.5, contradiction=0.0, neutral=0.5)] * len(pairs)

        # predict() accepts list of (str, str) tuples directly
        # apply_softmax=True → output probabilities sum to 1.0 per pair
        # shape: (N, num_classes) as numpy array
        raw_scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            apply_softmax=True,
        )

        schema = self._label_schema
        results: List[NLIResult] = []

        for row in raw_scores:
            # Use label schema to correctly index into the probability vector
            entailment   = float(row[schema["entailment"]])
            contradiction = float(row[schema["contradiction"]])
            neutral      = float(row[schema["neutral"]])
            results.append(NLIResult(
                entailment=entailment,
                contradiction=contradiction,
                neutral=neutral,
            ))

        return results

    def warmup(self) -> None:
        """Warm-up already performed in __init__."""
