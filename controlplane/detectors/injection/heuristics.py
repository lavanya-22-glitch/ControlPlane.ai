import re
from typing import List, Tuple

INJECTION_PATTERNS: List[Tuple[re.Pattern, float, str]] = [
    (
        re.compile(
            r"(?:ignore|disregard|forget|bypass|override)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules|directives|guidelines)",
            re.IGNORECASE,
        ),
        0.95,
        "Instruction override directive",
    ),
    (
        re.compile(
            r"(?:system\s+prompt\s+override|admin\s+override|developer\s+mode\s+enabled|jailbreak|DAN\s+mode)",
            re.IGNORECASE,
        ),
        0.90,
        "Jailbreak / system override phrase",
    ),
    (
        re.compile(
            r"(\[system\]|\[inst\]|<\|im_start\|>|<\|im_end\|>|<<sys>>|```system)",
            re.IGNORECASE,
        ),
        0.98,
        "Raw prompt delimiter injection",
    ),
    (
        re.compile(
            r"(?:you\s+are\s+now|act\s+as\s+(?:an\s+)?(?:unfiltered|unrestricted|evil|unaligned))",
            re.IGNORECASE,
        ),
        0.85,
        "Role-hijacking / unrestricted mode directive",
    ),
    (
        re.compile(
            r"(?:reveal|repeat|dump|print|output)\s+(?:your\s+)?(?:initial|original|exact|hidden)?\s*(?:system\s+prompt|instructions|system\s+rules)",
            re.IGNORECASE,
        ),
        0.90,
        "System prompt exfiltration attempt",
    ),
]


def evaluate_heuristics(text: str) -> Tuple[float, List[str]]:
    """Scan text against prompt injection heuristic patterns.
    Returns (highest_risk_score, matched_reasons).
    """
    if not text:
        return 0.0, []

    max_score = 0.0
    reasons: List[str] = []

    for pattern, score, label in INJECTION_PATTERNS:
        if pattern.search(text):
            max_score = max(max_score, score)
            reasons.append(label)

    return max_score, reasons
