"""
rag/pipeline.py
───────────────
Full RAG pipeline: hybrid retrieval → cross-encoder rerank → LLM generation.

Two LLM endpoints are wired up:
  - DirectPipeline  → Gemini API directly (no guardrails)
  - ProxyPipeline   → ControlPlane proxy (hallucination guard, PII redaction, bias check)

Both use the same retrieval stack so comparison is apples-to-apples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from langchain_core.documents import Document

from rag.config import (
    OPENAI_API_KEY,
    GEMINI_MODEL,
    DIRECT_URL,
    PROXY_BASE_URL,
    CONTROLPLANE_APP_ID,
    MAX_TOKENS,
    TEMPERATURE,
    RERANK_TOP_N,
    DENSE_TOP_K,
    SPARSE_TOP_K,
)
from rag.embedder import MiniLMEmbedder
from rag.vectorstore import load_faiss_index, get_all_docs
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CosineReranker


# ── Prompt template ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for TechFlow platform support.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, say "I don't have information about that."
Do NOT make up facts or use outside knowledge."""

def _build_prompt(question: str, context_chunks: List[str]) -> str:
    context_block = "\n\n---\n\n".join(context_chunks)
    return f"""Context:
{context_block}

Question: {question}

Answer:"""


# ── Response dataclass ─────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    retrieved_chunks: List[Document]
    reranked_chunks: List[Document]
    rerank_scores: List[float]
    # Proxy-specific fields (None for direct calls)
    proxy_action: Optional[str] = None
    grounding_score: Optional[str] = None
    violation_reason: Optional[str] = None
    pii_detected: Optional[str] = None
    http_status: int = 200
    error: Optional[str] = None


# ── Shared retrieval layer ─────────────────────────────────────────────────

class RetrievalStack:
    """One-time loaded retrieval stack shared by both pipelines."""

    _instance: Optional["RetrievalStack"] = None

    def __init__(self) -> None:
        self.embedder = MiniLMEmbedder()
        self.faiss_store = load_faiss_index(self.embedder)
        all_docs = get_all_docs(self.faiss_store)
        self.bm25 = BM25Retriever(all_docs)
        self.hybrid = HybridRetriever(self.faiss_store, self.bm25)
        self.reranker = CosineReranker()

    @classmethod
    def get(cls) -> "RetrievalStack":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def retrieve_and_rerank(
        self, query: str
    ) -> tuple[List[Document], List[Document], List[float]]:
        """Returns (all_retrieved_docs, reranked_docs, rerank_scores)."""
        hybrid_results = self.hybrid.retrieve(
            query,
            dense_top_k=DENSE_TOP_K,
            sparse_top_k=SPARSE_TOP_K,
        )
        candidate_docs = [doc for doc, _ in hybrid_results]

        reranked = self.reranker.rerank(query, candidate_docs, top_n=RERANK_TOP_N)
        reranked_docs = [doc for doc, _ in reranked]
        reranked_scores = [score for _, score in reranked]

        return candidate_docs, reranked_docs, reranked_scores


# ── Direct pipeline ────────────────────────────────────────────────────────

class DirectPipeline:
    """
    Calls Gemini directly — no proxy, no guardrails.
    Used as the BEFORE baseline.
    """

    def __init__(self, stack: RetrievalStack | None = None) -> None:
        self._stack = stack or RetrievalStack.get()
        self._client = httpx.Client(timeout=60.0)

    def ask(self, question: str) -> RAGResponse:
        retrieved, reranked, scores = self._stack.retrieve_and_rerank(question)
        context_chunks = [d.page_content for d in reranked]
        user_content = _build_prompt(question, context_chunks)

        payload = {
            "model": GEMINI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        t0 = time.perf_counter()
        try:
            resp = self._client.post(
                f"{DIRECT_URL}chat/completions",
                json=payload,
                headers=headers,
            )
            latency = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            return RAGResponse(
                answer=data["choices"][0]["message"]["content"],
                latency_ms=latency,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                retrieved_chunks=retrieved,
                reranked_chunks=reranked,
                rerank_scores=scores,
                http_status=resp.status_code,
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            return RAGResponse(
                answer="",
                latency_ms=latency,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                retrieved_chunks=retrieved,
                reranked_chunks=reranked,
                rerank_scores=scores,
                error=str(exc),
                http_status=getattr(getattr(exc, "response", None), "status_code", 0),
            )


# ── Proxy pipeline ─────────────────────────────────────────────────────────

class ProxyPipeline:
    """
    Calls the ControlPlane proxy — with hallucination guard, PII redaction,
    bias detection, token tracking, and latency overhead.
    Used as the AFTER showcase.
    """

    def __init__(self, stack: RetrievalStack | None = None) -> None:
        self._stack = stack or RetrievalStack.get()
        self._client = httpx.Client(timeout=60.0)

    def ask(self, question: str) -> RAGResponse:
        retrieved, reranked, scores = self._stack.retrieve_and_rerank(question)
        context_chunks = [d.page_content for d in reranked]
        user_content = _build_prompt(question, context_chunks)

        payload = {
            "model": GEMINI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            # ControlPlane-specific: send raw context for NLI grounding check
            "retrieved_context": context_chunks,
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "X-ControlPlane-App-ID": CONTROLPLANE_APP_ID,
        }

        t0 = time.perf_counter()
        try:
            resp = self._client.post(
                f"{PROXY_BASE_URL}chat/completions",
                json=payload,
                headers=headers,
            )
            latency = (time.perf_counter() - t0) * 1000

            # Read telemetry headers injected by the proxy
            action          = resp.headers.get("x-controlplane-policy-action", "ALLOW")
            grounding_score = resp.headers.get("x-controlplane-score-grounding")
            violation       = resp.headers.get("x-controlplane-violation-reason")
            pii_detected    = resp.headers.get("x-controlplane-pii-detected")

            answer = ""
            usage  = {}
            if resp.status_code == 200:
                data   = resp.json()
                usage  = data.get("usage", {})
                answer = data["choices"][0]["message"]["content"]

            return RAGResponse(
                answer=answer,
                latency_ms=latency,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                retrieved_chunks=retrieved,
                reranked_chunks=reranked,
                rerank_scores=scores,
                proxy_action=action,
                grounding_score=grounding_score,
                violation_reason=violation,
                pii_detected=pii_detected,
                http_status=resp.status_code,
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            return RAGResponse(
                answer="",
                latency_ms=latency,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                retrieved_chunks=retrieved,
                reranked_chunks=reranked,
                rerank_scores=scores,
                error=str(exc),
                http_status=0,
            )
