import re
from typing import List, Tuple


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def align_claims_to_context(claims: List[str], context_chunks: List[str]) -> List[Tuple[str, str, float]]:
    """Align each claim to its most relevant retrieved context chunk.
    Returns list of (best_context_chunk, claim, overlap_ratio).
    """
    if not context_chunks:
        return [("", claim, 0.0) for claim in claims]

    aligned_pairs: List[Tuple[str, str, float]] = []

    for claim in claims:
        claim_tokens = _tokenize(claim)
        if not claim_tokens:
            aligned_pairs.append((context_chunks[0], claim, 0.0))
            continue

        best_chunk = context_chunks[0]
        max_overlap = 0.0

        for chunk in context_chunks:
            chunk_tokens = _tokenize(chunk)
            if not chunk_tokens:
                continue
            intersection = claim_tokens.intersection(chunk_tokens)
            overlap = len(intersection) / len(claim_tokens)
            if overlap > max_overlap:
                max_overlap = overlap
                best_chunk = chunk

        aligned_pairs.append((best_chunk, claim, max_overlap))

    return aligned_pairs
