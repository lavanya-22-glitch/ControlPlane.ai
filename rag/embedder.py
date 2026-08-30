"""
rag/embedder.py
───────────────
Lightweight ONNX-based embedder — completely torch-free.
Uses HuggingFace Optimum + ONNX Runtime to run all-MiniLM-L6-v2.

Why ONNX and not sentence-transformers?
  sentence-transformers → imports transformers → imports torch → blocked by
  Windows Application Control DLL policy on this machine.
  Optimum exports the model to ONNX once, then runs via onnxruntime (pure C++).
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings

from rag.config import EMBED_MODEL_NAME, EMBED_BATCH_SIZE

# Lazy imports
_ort_session = None
_tokenizer = None

def _load_models():
    """Download & cache the ONNX model on first call."""
    global _ort_session, _tokenizer
    if _ort_session is not None:
        return

    print(f"[Embedder] Loading ONNX model '{EMBED_MODEL_NAME}' ...")

    # Import inside function to keep it lazy
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    import onnxruntime as ort

    repo_id = f"Xenova/{EMBED_MODEL_NAME}"
    
    # Download model and tokenizer
    model_path = hf_hub_download(repo_id=repo_id, filename="onnx/model.onnx")
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")

    _tokenizer = Tokenizer.from_file(tokenizer_path)
    # Enable truncation and padding
    _tokenizer.enable_truncation(max_length=256)
    _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    _ort_session = ort.InferenceSession(model_path)
    print("[Embedder] ONNX model ready.")


def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool over non-padding token positions."""
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    return summed / counts


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
    return vecs / norms


def _embed_batch(texts: List[str]) -> np.ndarray:
    _load_models()
    
    encoded = _tokenizer.encode_batch(texts)
    
    input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)
    
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }
    
    outputs = _ort_session.run(None, ort_inputs)
    # outputs[0] is last_hidden_state
    pooled = _mean_pool(outputs[0], attention_mask)
    return _normalize(pooled)


class MiniLMEmbedder(Embeddings):
    """LangChain-compatible embedder backed by ONNX Runtime (torch-free)."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            all_vecs.append(_embed_batch(batch))
        return np.vstack(all_vecs).tolist()

    def embed_query(self, text: str) -> List[float]:
        return _embed_batch([text])[0].tolist()

