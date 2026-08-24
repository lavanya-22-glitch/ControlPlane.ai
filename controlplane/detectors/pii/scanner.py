import re
import logging
from typing import List, Dict, Any, Tuple, Optional

from controlplane.detectors.base import BaseDetector, GuardStage, DetectionResult, GuardContext
from controlplane.detectors.pii.vault import pii_vault
from controlplane.pdp.types import PDPAction, ViolationCode

logger = logging.getLogger("controlplane.detectors.pii")

# Fast-path compiled regular expressions (<5ms)
REGEX_PATTERNS: Dict[str, re.Pattern] = {
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
}

# Optional fallback for names/contextual entities
NAME_PATTERN = re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")


class PIIScannerGuard(BaseDetector):
    """Dual-Tier Reversible PII Detector and Masker."""

    def __init__(self):
        self._presidio_analyzer = None
        self._init_presidio_if_available()

    def _init_presidio_if_available(self):
        try:
            from presidio_analyzer import AnalyzerEngine
            self._presidio_analyzer = AnalyzerEngine()
            logger.info("Presidio Analyzer engine initialized successfully for Tier-2 PII.")
        except Exception as e:
            logger.info("Presidio not loaded (%s). Using high-speed Regex + Heuristics.", e)

    @property
    def name(self) -> str:
        return "pii_scanner"

    @property
    def stage(self) -> GuardStage:
        return GuardStage.PRE_EXECUTION

    def _scan_text(self, text: str, allowed_entities: List[str]) -> List[Tuple[str, str, int, int]]:
        """Find entity matches in text. Returns list of (entity_type, raw_value, start_idx, end_idx)."""
        matches: List[Tuple[str, str, int, int]] = []
        is_all = "ALL" in allowed_entities

        # 1. Tier-1 Fast Regex Scan
        for entity_type, pattern in REGEX_PATTERNS.items():
            if is_all or entity_type in allowed_entities:
                for match in pattern.finditer(text):
                    matches.append((entity_type, match.group(0), match.start(), match.end()))

        # 2. Tier-2 Contextual / NLP Scan
        if is_all or "PERSON" in allowed_entities:
            if self._presidio_analyzer:
                try:
                    results = self._presidio_analyzer.analyze(text=text, entities=["PERSON"], language="en")
                    for r in results:
                        val = text[r.start:r.end]
                        matches.append(("PERSON", val, r.start, r.end))
                except Exception as e:
                    logger.debug("Presidio scan exception: %s", e)
            else:
                for match in NAME_PATTERN.finditer(text):
                    matches.append(("PERSON", match.group(0), match.start(), match.end()))

        # Sort matches by start index descending to allow in-place substitution
        matches.sort(key=lambda m: m[2], reverse=True)
        return matches

    async def evaluate(self, ctx: GuardContext) -> DetectionResult:
        pii_config = ctx.policy.pre_execution
        action_mode = pii_config.pii_action  # "mask" | "block" | "none"
        if action_mode == "none":
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.ALLOW,
            )

        target_entities = pii_config.masked_entities
        all_detected_entities: List[str] = []
        total_tokens_masked = 0

        # Scan and transform each message in the prompt history
        for msg in ctx.messages:
            content = msg.get("content", "")
            if not isinstance(content, str) or not content:
                continue

            matches = self._scan_text(content, target_entities)
            if not matches:
                continue

            # If policy says "block" when PII is present:
            if action_mode == "block":
                entities = list(set(m[0] for m in matches))
                return DetectionResult(
                    detector_name=self.name,
                    stage=self.stage,
                    passed=False,
                    suggested_action=PDPAction.BLOCK,
                    violation_code=ViolationCode.PII_DISALLOWED,
                    reason=f"PII entities detected ({', '.join(entities)}) and policy forbids PII.",
                    metadata={"entities": entities},
                )

            # In "mask" mode, perform reversible tokenization
            modified_text = content
            for entity_type, raw_val, start, end in matches:
                all_detected_entities.append(entity_type)
                token = pii_vault.get_or_create_token(ctx.trace_id, entity_type, raw_val)
                modified_text = modified_text[:start] + token + modified_text[end:]
                total_tokens_masked += 1

            msg["content"] = modified_text

        unique_entities = list(set(all_detected_entities))
        ctx.pii_entities_detected.extend(unique_entities)

        if total_tokens_masked > 0:
            return DetectionResult(
                detector_name=self.name,
                stage=self.stage,
                passed=True,
                suggested_action=PDPAction.TRANSFORM,
                metadata={
                    "masked_count": total_tokens_masked,
                    "entities": unique_entities,
                },
            )

        return DetectionResult(
            detector_name=self.name,
            stage=self.stage,
            passed=True,
            suggested_action=PDPAction.ALLOW,
        )
