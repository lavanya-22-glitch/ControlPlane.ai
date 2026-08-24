import re
from typing import List


def split_claims(text: str) -> List[str]:
    """Split completion text into discrete checkable factual claims/sentences."""
    if not text:
        return []

    # Clean markdown headers and bullet points
    cleaned = re.sub(r"^\s*[-*#]+\s*", "", text, flags=re.MULTILINE)

    # Split by sentence boundaries (. ! ? \n)
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    claims: List[str] = []

    for s in raw_sentences:
        s_clean = s.strip()
        # Filter out very short phrases, greetings, or conversational noise
        if len(s_clean.split()) >= 3:
            claims.append(s_clean)

    return claims if claims else [text.strip()]
