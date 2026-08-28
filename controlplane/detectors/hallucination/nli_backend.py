"""
NLI Backend abstraction layer for RAG grounding verification.

Architecture:
  NLIBackend (ABC)
    ├── CrossEncoderNLIBackend  — sentence-transformers CrossEncoder, joint cross-attention
    ├── TransformersNLIBackend  — HuggingFace pipeline fallback, batched, thread-pooled
    ├── ONNXNLIBackend          — ONNX Runtime INT8 session, minimal overhead
    └── HeuristicNLIBackend     — Lexical overlap + negation (always-available fast-path)

  auto_select_backend()         — Environment-aware factory with warm-up

Selection priority ("auto" mode):
  1. ONNX         — if onnx_model_path is configured
  2. CrossEncoder — if sentence_transformers is installed  (recommended)
  3. Transformers — if transformers + torch are installed (pipeline fallback)
  4. Heuristic    — always available, zero dependencies

Latency budget targets:
  HeuristicNLIBackend     : <5ms   per batch
  CrossEncoderNLIBackend  : <40ms  per batch (CPU, max_length=128, MiniLM)
  TransformersNLIBackend  : <250ms per batch (CPU, fp32, max_length=256)
  ONNXNLIBackend          : <50ms  per batch (INT8 quantised)
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("controlplane.detectors.hallucination.nli_backend")

# ---------------------------------------------------------------------------
# Shared thread pool — transformer inference must never block the event loop
# ---------------------------------------------------------------------------
_INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nli_infer")


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NLIResult:
    """Probabilities for a single (premise, hypothesis) pair."""
    entailment: float
    contradiction: float
    neutral: float

    @property
    def grounding_score(self) -> float:
        """Entailment − Contradiction clamped to [0, 1].

        Positive = supported by context.
        Negative = contradicted by context (clamped to 0.0).
        """
        return float(np.clip(self.entailment - self.contradiction, 0.0, 1.0))

    def __repr__(self) -> str:
        return (
            f"NLIResult(E={self.entailment:.3f}, C={self.contradiction:.3f}, "
            f"N={self.neutral:.3f}, grounding={self.grounding_score:.3f})"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class NLIBackend(ABC):
    """Abstract NLI scoring backend.

    Contract:
        - ``score_pairs`` receives a list of (premise/context, hypothesis/claim) tuples.
        - Returns an ``NLIResult`` per pair in the *same order*.
        - Must be thread-safe; may be called concurrently.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...

    @abstractmethod
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """Synchronous batch scorer — runs in the thread pool executor."""
        ...

    async def async_score_pairs(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """Offload synchronous ``score_pairs`` to the shared thread pool
        so it never blocks the asyncio event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_INFERENCE_EXECUTOR, self.score_pairs, pairs)

    def warmup(self) -> None:
        """Optional warmup — run a dummy inference to load model weights
        into memory so the first real request is fast."""


# ---------------------------------------------------------------------------
# Heuristic backend (always available, <5ms)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


class HeuristicNLIBackend(NLIBackend):
    """Pure-Python lexical overlap NLI approximation.

    Scoring:
      overlap_ratio  = |claim_tokens ∩ context_tokens| / |claim_tokens|
      negation_penalty = −0.35 if unmatched negation detected
      entailment  = clip(overlap * 1.25, 0, 1)
      contradiction = clip(negation_penalty magnitude, 0, 0.6)
      neutral     = 1 − entailment − contradiction
    """

    _NEGATION_RE = re.compile(
        r"\b(not|never|no|cannot|can't|won't|isn't|aren't|wasn't|weren't|doesn't|don't|didn't)\b",
        re.IGNORECASE,
    )

    @property
    def backend_name(self) -> str:
        return "heuristic"

    def _has_negation(self, text: str) -> bool:
        return bool(self._NEGATION_RE.search(text))

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        results: List[NLIResult] = []
        for context, claim in pairs:
            claim_tokens = _tokenize(claim)
            if not claim_tokens:
                results.append(NLIResult(entailment=0.5, contradiction=0.0, neutral=0.5))
                continue

            ctx_tokens = _tokenize(context)
            overlap = len(claim_tokens & ctx_tokens) / len(claim_tokens)

            # Unmatched negation: claim negates something the context affirms (or vice-versa)
            negation_mismatch = (
                self._has_negation(claim) and not self._has_negation(context)
            ) or (
                not self._has_negation(claim) and self._has_negation(context)
            )

            raw_entailment = min(1.0, overlap * 1.25)
            contradiction = 0.35 if negation_mismatch and overlap < 0.4 else 0.0
            entailment = max(0.0, raw_entailment - contradiction)
            neutral = max(0.0, 1.0 - entailment - contradiction)
            results.append(NLIResult(entailment=entailment, contradiction=contradiction, neutral=neutral))
        return results


# ---------------------------------------------------------------------------
# Transformers backend (requires: transformers, torch)
# ---------------------------------------------------------------------------

class TransformersNLIBackend(NLIBackend):
    """HuggingFace cross-encoder NLI model backend.

    Warm-up: model is loaded once at construction; subsequent calls go directly
    to the cached model/tokenizer in memory.

    Batching: all (context, claim) pairs are fed as a single batch to minimise
    tokenization + inference overhead (one forward pass per request).

    Thread-safety: the model is loaded in eval() mode with torch.no_grad()
    globally set during __init__; inference executes in the shared executor.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        max_length: int = 256,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._pipeline = None
        self._label_map: dict = {}
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline
            import torch

            device = 0 if torch.cuda.is_available() else -1  # CPU fallback
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self._model_name,
                device=device,
                top_k=None,          # return all label scores
                truncation=True,
                max_length=self._max_length,
            )
            # Warm up: single dummy pair so tokenizer + weights are JIT-compiled
            self._pipeline([{"text": "warmup context", "text_pair": "warmup claim"}])
            logger.info(
                "TransformersNLIBackend loaded model='%s' on device=%s",
                self._model_name,
                "cuda" if device == 0 else "cpu",
            )
        except Exception as exc:
            logger.error("Failed to load Transformers NLI model: %s", exc)
            raise

    @property
    def backend_name(self) -> str:
        return f"transformers:{self._model_name}"

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        if not pairs:
            return []

        # Build HuggingFace pipeline input format
        hf_inputs = [{"text": ctx, "text_pair": claim} for ctx, claim in pairs]

        results: List[NLIResult] = []
        # Process in sub-batches to keep memory bounded
        for i in range(0, len(hf_inputs), self._batch_size):
            chunk = hf_inputs[i : i + self._batch_size]
            raw_outputs = self._pipeline(chunk)  # list[list[{label, score}]]
            for label_scores in raw_outputs:
                scores = {item["label"].lower(): item["score"] for item in label_scores}
                results.append(
                    NLIResult(
                        entailment=scores.get("entailment", 0.0),
                        contradiction=scores.get("contradiction", 0.0),
                        neutral=scores.get("neutral", 0.0),
                    )
                )
        return results

    def warmup(self) -> None:
        """Already done in __init__; exposed for interface compliance."""


# ---------------------------------------------------------------------------
# ONNX backend (requires: onnxruntime, tokenizers)
# ---------------------------------------------------------------------------

class ONNXNLIBackend(NLIBackend):
    """ONNX Runtime INT8 cross-encoder NLI backend.

    Requires a pre-exported ONNX model and its fast tokenizer.
    Target latency: <50ms per batch on CPU (INT8 quantized).
    """

    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None, max_length: int = 256) -> None:
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path or model_path
        self._max_length = max_length
        self._session = None
        self._tokenizer = None
        self._id2label: dict = {}
        self._load()

    def _load(self) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = Tokenizer.from_pretrained(self._tokenizer_path)
            self._tokenizer.enable_truncation(max_length=self._max_length)
            self._tokenizer.enable_padding()

            # Infer label ordering from model metadata
            meta = self._session.get_modelmeta().custom_metadata_map
            label_str = meta.get("id2label", '{"0":"contradiction","1":"neutral","2":"entailment"}')
            import json
            self._id2label = {int(k): v.lower() for k, v in json.loads(label_str).items()}

            logger.info("ONNXNLIBackend loaded model from '%s'", self._model_path)
        except Exception as exc:
            logger.error("Failed to load ONNX NLI model: %s", exc)
            raise

    @property
    def backend_name(self) -> str:
        return f"onnx:{self._model_path}"

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        import numpy as np

        if not pairs or self._session is None:
            return [NLIResult(entailment=0.5, contradiction=0.0, neutral=0.5)] * len(pairs)

        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        encodings = self._tokenizer.encode_batch(list(zip(premises, hypotheses)))

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        logits = self._session.run(None, ort_inputs)[0]  # shape: (N, num_labels)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        results: List[NLIResult] = []
        for row in probs:
            label_probs = {self._id2label.get(i, str(i)): float(row[i]) for i in range(len(row))}
            results.append(
                NLIResult(
                    entailment=label_probs.get("entailment", 0.0),
                    contradiction=label_probs.get("contradiction", 0.0),
                    neutral=label_probs.get("neutral", 0.0),
                )
            )
        return results


# ---------------------------------------------------------------------------
# Factory: auto-select best available backend
# ---------------------------------------------------------------------------

_SINGLETON_BACKEND: Optional[NLIBackend] = None


def auto_select_backend(
    preferred: str = "auto",
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    onnx_model_path: Optional[str] = None,
) -> NLIBackend:
    """Return and cache the best available NLI backend.

    Selection priority (when ``preferred="auto"``):
      1. ONNX         — if ``onnx_model_path`` is provided
      2. CrossEncoder — if ``sentence_transformers`` is importable  (preferred)
      3. Transformers — if ``transformers`` + ``torch`` are importable
      4. Heuristic    — always available, no dependencies

    The result is cached in a module-level singleton so model weights
    are loaded exactly once per process lifetime.

    Args:
        preferred: Backend name: ``"auto"`` | ``"cross_encoder"`` | ``"onnx"``
                   | ``"transformers"`` | ``"heuristic"``.
        model_name: HuggingFace model ID for cross-encoder or transformers backends.
        onnx_model_path: Path to a quantised ONNX model file (enables ONNX backend).
    """
    global _SINGLETON_BACKEND

    # Return cached singleton for auto mode (model loaded once per process)
    if _SINGLETON_BACKEND is not None and preferred == "auto":
        return _SINGLETON_BACKEND

    backend: NLIBackend

    # ---- 1. ONNX (explicit path required) ----------------------------------
    if preferred == "onnx" or (preferred == "auto" and onnx_model_path):
        if not onnx_model_path:
            raise ValueError("onnx_model_path must be provided when preferred='onnx'")
        try:
            backend = ONNXNLIBackend(model_path=onnx_model_path)
            logger.info("NLI backend selected: ONNX (%s)", onnx_model_path)
        except Exception as exc:
            logger.warning("ONNX backend unavailable (%s); continuing auto-selection.", exc)
            preferred = "auto"  # fall through to next tier
            onnx_model_path = None

    # ---- 2. CrossEncoder (sentence-transformers) — preferred if available --
    if preferred in ("auto", "cross_encoder"):
        try:
            # Import lazily to avoid hard dependency
            from controlplane.detectors.hallucination.cross_encoder import CrossEncoderNLIBackend  # noqa
            backend = CrossEncoderNLIBackend(model_name=model_name)
            logger.info("NLI backend selected: CrossEncoder (%s)", model_name)
        except Exception as exc:
            if preferred == "cross_encoder":
                # Explicit request for cross_encoder — don't silently fall back
                logger.error("CrossEncoder backend requested but unavailable: %s", exc)
                raise
            logger.info(
                "CrossEncoder backend unavailable (%s); trying Transformers.", exc
            )

            # ---- 3. Transformers pipeline fallback -------------------------
            try:
                import transformers  # noqa: F401
                import torch          # noqa: F401
                backend = TransformersNLIBackend(model_name=model_name)
                logger.info("NLI backend selected: Transformers (%s)", model_name)
            except Exception as exc2:
                logger.warning(
                    "Transformers backend unavailable (%s); falling back to Heuristic.", exc2
                )
                # ---- 4. Heuristic — always available ----------------------
                backend = HeuristicNLIBackend()
                logger.info("NLI backend selected: Heuristic (fallback)")

    elif preferred == "transformers":
        try:
            import transformers  # noqa: F401
            import torch          # noqa: F401
            backend = TransformersNLIBackend(model_name=model_name)
            logger.info("NLI backend selected: Transformers (%s)", model_name)
        except Exception as exc:
            logger.warning("Transformers backend unavailable (%s); falling back to Heuristic.", exc)
            backend = HeuristicNLIBackend()

    else:  # "heuristic" or unknown
        backend = HeuristicNLIBackend()
        logger.info("NLI backend selected: Heuristic")

    if preferred == "auto":
        _SINGLETON_BACKEND = backend

    return backend
