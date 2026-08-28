"""
Citation Marker Parser for LLM Completion Text.

Parses inline citation markers from LLM-generated completions and
extracts the text span (preceding sentence) that each citation is
meant to support.

Supported citation formats:
  [1]                  — Numeric citation
  [1, 2], [1,2,3]     — Multi-reference numeric
  [Doc A]              — Named document citation
  [Source: Title]      — Source-prefixed citation
  [Ref: HR-Policy-01]  — Reference-prefixed citation
  (Smith et al., 2023) — Author-year citation (academic)
  (Author 2023)        — Author-year compact

Extraction strategy:
  For each citation marker found in the text, the "preceding sentence"
  — the sentence immediately before the marker — is extracted as the
  claimed text that the citation is meant to support.

  This is the text we then compare to retrieved context chunks in the
  citation similarity check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Citation:
    """A single parsed citation marker and the claim text it supports."""
    marker: str            # Full marker text: "[1]", "[Doc A]", "(Smith et al.)"
    marker_type: str       # "numeric" | "named" | "source" | "author_year"
    marker_value: str      # Inner value: "1", "Doc A", "Smith et al., 2023"
    preceding_text: str    # The sentence/phrase this citation is annotating
    position: int          # Char offset of the marker start in the original text

    @property
    def is_multi_reference(self) -> bool:
        """True for [1, 2, 3] style multi-citations."""
        return self.marker_type == "numeric" and "," in self.marker_value

    def __repr__(self) -> str:
        preview = self.preceding_text[:50].replace("\n", " ")
        return f'Citation({self.marker!r} ← "{preview}...")'


# ---------------------------------------------------------------------------
# Compiled regex patterns (priority order matters — more specific first)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# Sentence terminator used to extract preceding context
_SENT_END_RE = re.compile(r'[.!?]\s*$')

_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # [Source: Title] or [Ref: name] — most specific, must come first
    (
        "source",
        re.compile(
            r'\[(?:Source|Ref|Reference|Doc|Document):\s*([^\]]{1,80})\]',
            re.IGNORECASE,
        ),
    ),
    # [1], [2,3], [1, 2, 3] — pure numeric
    (
        "numeric",
        re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]'),
    ),
    # [Doc A], [Policy-2024], [HR-001] — named with ≥2 chars, alphanumeric
    (
        "named",
        re.compile(r'\[([A-Za-z][A-Za-z0-9\s\-_.]{1,40})\]'),
    ),
    # (Smith et al., 2023) or (Author, 2023)
    (
        "author_year",
        re.compile(
            r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?(?:,?\s*\d{4})?)\)',
        ),
    ),
]


# ---------------------------------------------------------------------------
# Sentence extraction utilities
# ---------------------------------------------------------------------------

def _extract_preceding_sentence(text: str, marker_start: int, max_chars: int = 300) -> str:
    """Extract the sentence immediately preceding a citation marker.

    Strategy:
      1. Slice text up to the marker position.
      2. Split by sentence boundaries.
      3. Return the last non-empty sentence.
      4. If the text has no sentence boundaries, return up to ``max_chars``
         of text before the marker.
    """
    text_before = text[:marker_start].strip()
    if not text_before:
        return ""

    # Split on sentence boundaries
    sentences = _SENTENCE_SPLIT_RE.split(text_before)
    # Walk backwards to find the last non-trivial sentence
    for sent in reversed(sentences):
        sent = sent.strip()
        if len(sent.split()) >= 3:  # ignore very short fragments
            return sent[-max_chars:]  # cap length

    # Fallback: return last max_chars of text before marker
    return text_before[-max_chars:]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class CitationParser:
    """Parses inline citation markers from LLM completion text.

    Usage::
        parser = CitationParser()
        citations = parser.parse("Refunds take 7 days [1]. Employees get 30 days leave [2].")
        # citations[0].marker == "[1]", preceding_text == "Refunds take 7 days"
        # citations[1].marker == "[2]", preceding_text == "Employees get 30 days leave"
    """

    def __init__(self, max_preceding_chars: int = 300) -> None:
        self._max_chars = max_preceding_chars

    def parse(self, text: str) -> List[Citation]:
        """Find all citation markers and their supporting text spans.

        Args:
            text: LLM completion text to parse.

        Returns:
            List of ``Citation`` objects in document order (by position).
        """
        if not text:
            return []

        found: List[Citation] = []
        seen_positions: set = set()

        for marker_type, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                start = match.start()
                # Avoid double-counting positions already claimed by a higher-priority pattern
                if any(abs(start - p) < 3 for p in seen_positions):
                    continue

                marker_value = match.group(1).strip()
                preceding = _extract_preceding_sentence(text, start, self._max_chars)

                if not preceding:
                    # No preceding text — citation at very start of text, skip
                    continue

                found.append(Citation(
                    marker=match.group(0),
                    marker_type=marker_type,
                    marker_value=marker_value,
                    preceding_text=preceding,
                    position=start,
                ))
                seen_positions.add(start)

        # Sort by document position
        found.sort(key=lambda c: c.position)
        return found

    def has_citations(self, text: str) -> bool:
        """Quick check — returns True if any citation marker is present."""
        return any(p.search(text) for _, p in _PATTERNS)
