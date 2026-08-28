"""
Context Aligner: maps each factual claim to its most relevant RAG context chunk.

Alignment strategy:
    For each claim, compute Jaccard-style token overlap against every context
    chunk and select the chunk with the highest overlap ratio.

    This is intentionally lightweight (pure Python regex tokenization) because
    it runs *before* NLI inference and must complete in <5ms regardless of the
    number of claims × chunks.

Returns:
    List[ClaimContextPair] — typed contracts used by the NLI verifier.
"""

from __future__ import annotations

import re
from typing import List

from controlplane.detectors.hallucination.types import ClaimContextPair


def _tokenize(text: str) -> frozenset:
    """Lowercase word tokenization. Returns frozenset for O(1) intersection."""
    return frozenset(re.findall(r"\w+", text.lower()))


def align_claims_to_context(
    claims: List[str],
    context_chunks: List[str],
) -> List[ClaimContextPair]:
    """Align each claim to the single best-matching context chunk.

    Args:
        claims: Extracted factual sentences from the LLM completion.
        context_chunks: RAG-retrieved context passages.

    Returns:
        One ``ClaimContextPair`` per claim, preserving input order.
    """
    if not context_chunks:
        return [
            ClaimContextPair(claim=c, context_chunk="", lexical_overlap=0.0)
            for c in claims
        ]

    # Pre-tokenise all context chunks once — O(C) instead of O(N×C)
    tokenised_chunks: List[frozenset] = [_tokenize(chunk) for chunk in context_chunks]

    aligned: List[ClaimContextPair] = []

    for claim in claims:
        claim_tokens = _tokenize(claim)

        if not claim_tokens:
            aligned.append(
                ClaimContextPair(
                    claim=claim,
                    context_chunk=context_chunks[0],
                    lexical_overlap=0.0,
                )
            )
            continue

        best_chunk = context_chunks[0]
        best_overlap = 0.0

        for chunk_text, chunk_tokens in zip(context_chunks, tokenised_chunks):
            if not chunk_tokens:
                continue
            # Overlap ratio = |claim ∩ chunk| / |claim|  (claim-centric recall)
            overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk = chunk_text

        aligned.append(
            ClaimContextPair(
                claim=claim,
                context_chunk=best_chunk,
                lexical_overlap=best_overlap,
            )
        )

    return aligned
