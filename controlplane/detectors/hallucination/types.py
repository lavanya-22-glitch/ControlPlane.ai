"""
Typed data contract for claim-context alignment.

Separating into its own module keeps the aligner, verifier, and backend
decoupled — each can be imported independently without circular deps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClaimContextPair:
    """A single factual claim paired with its best-matching RAG context chunk."""
    claim: str
    context_chunk: str
    lexical_overlap: float  # [0, 1] — used for pre-filter gating
