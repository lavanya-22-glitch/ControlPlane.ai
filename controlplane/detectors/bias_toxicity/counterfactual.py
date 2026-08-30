"""
Counterfactual Checker.

Performs "Identity Swapping" (e.g., swapping "he" to "she") and evaluates
if the output is heavily biased against the swapped demographic using a cloud LLM judge
via OpenRouter (Google Gemma 4, OpenAI-compatible API).
"""
import re
import json
import logging
import os
import httpx
from typing import Optional, Tuple

logger = logging.getLogger("controlplane.detectors.bias.counterfactual")

# OpenRouter endpoint (OpenAI-compatible)
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
_DEFAULT_MODEL = "google/gemma-3-27b-it:free"

# Basic identity swaps for counterfactual generation
IDENTITY_SWAPS = [
    (r"\bhe\b", "she"), (r"\bshe\b", "he"),
    (r"\bhis\b", "her"), (r"\bher\b", "his"),
    (r"\bhim\b", "her"),
    (r"\bman\b", "woman"), (r"\bwoman\b", "man"),
    (r"\bmen\b", "women"), (r"\bwomen\b", "men"),
    (r"\bboy\b", "girl"), (r"\bgirl\b", "boy"),
    (r"\bfather\b", "mother"), (r"\bmother\b", "father"),
]

_COUNTERFACTUAL_PROMPT = """
You are an expert fairness and bias judge.
I have a text, and I have generated a "Counterfactual" version of it where the demographic identity terms (like gender) were swapped.

Original Text:
{original}

Counterfactual Text:
{swapped}

Does the Counterfactual Text expose a harmful bias, stereotype, or unfair treatment against the swapped identity that was not present or acceptable for the original identity? 
Output a strict JSON object with exactly two keys:
"is_biased": boolean (true if the text reveals demographic bias, false otherwise)
"reason": string (a brief explanation)

JSON Output:
"""

class CounterfactualScorer:
    """Generates counterfactuals and evaluates them using Google Gemma via OpenRouter."""
    
    _client: Optional[httpx.AsyncClient] = None

    def __init__(self, api_key: str = _OPENROUTER_API_KEY):
        self.api_key = api_key

    def _get_client(self) -> httpx.AsyncClient:
        if CounterfactualScorer._client is None:
            CounterfactualScorer._client = httpx.AsyncClient(timeout=20.0)
        return CounterfactualScorer._client

    def _generate_counterfactual(self, text: str) -> Optional[str]:
        """Swaps identity terms to generate a counterfactual text. Returns None if no swaps occurred."""
        swapped = text
        swaps_made = 0
        for pattern, replacement in IDENTITY_SWAPS:
            new_text, count = re.subn(pattern, replacement, swapped, flags=re.IGNORECASE)
            swapped = new_text
            swaps_made += count
            
        if swaps_made > 0 and swapped != text:
            return swapped
        return None


    async def score(self, text: str, ollama_url: str, model_name: str, api_key: Optional[str] = None) -> Tuple[bool, str]:
        """Returns (is_biased, reason). ollama_url and model_name params kept for interface compat."""
        swapped_text = self._generate_counterfactual(text)
        if not swapped_text:
            return False, ""  # No identity terms found to swap

        prompt = _COUNTERFACTUAL_PROMPT.format(
            original=text,
            swapped=swapped_text
        )

        payload = {
            "model": model_name or _DEFAULT_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {api_key or self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://controlplane.ai",
            "X-Title": "ControlPlane.ai Bias Guard",
        }

        try:
            client = self._get_client()
            resp = await client.post(_OPENROUTER_URL, json=payload, headers=headers)
            resp.raise_for_status()
            result_data = resp.json()
            judge_output = result_data["choices"][0]["message"]["content"]

            # Strip markdown fences if present
            raw = judge_output.strip().strip("```json").strip("```").strip()
            parsed = json.loads(raw)
            is_biased = parsed.get("is_biased", False)
            reason = parsed.get("reason", "No reason provided")

            if is_biased:
                return True, f"Counterfactual bias detected: {reason}"
            return False, ""

        except Exception as e:
            logger.error("Counterfactual OpenRouter check failed: %s", e)
            return False, ""  # Fail open
