import threading
from typing import Dict, Tuple, Optional


class SessionPIIVault:
    """Thread-safe in-memory session token vault for reversible PII masking."""

    def __init__(self):
        self._lock = threading.Lock()
        # trace_id -> { raw_value: token }
        self._forward_map: Dict[str, Dict[str, str]] = {}
        # trace_id -> { token: raw_value }
        self._reverse_map: Dict[str, Dict[str, str]] = {}
        # trace_id -> { entity_type: counter }
        self._counters: Dict[str, Dict[str, int]] = {}

    def get_or_create_token(self, trace_id: str, entity_type: str, raw_value: str) -> str:
        """Retrieve existing token for raw_value or generate a new deterministic placeholder."""
        with self._lock:
            if trace_id not in self._forward_map:
                self._forward_map[trace_id] = {}
                self._reverse_map[trace_id] = {}
                self._counters[trace_id] = {}

            if raw_value in self._forward_map[trace_id]:
                return self._forward_map[trace_id][raw_value]

            # Generate new token
            count = self._counters[trace_id].get(entity_type, 0) + 1
            self._counters[trace_id][entity_type] = count
            token = f"<{entity_type.upper()}_{count}>"

            self._forward_map[trace_id][raw_value] = token
            self._reverse_map[trace_id][token] = raw_value
            return token

    def get_raw_value(self, trace_id: str, token: str) -> Optional[str]:
        """Look up raw sensitive string from token placeholder."""
        with self._lock:
            return self._reverse_map.get(trace_id, {}).get(token)

    def get_all_mappings(self, trace_id: str) -> Dict[str, str]:
        """Return token -> raw_value dictionary for a given trace."""
        with self._lock:
            return dict(self._reverse_map.get(trace_id, {}))

    def clear(self, trace_id: str) -> None:
        """Evict session data for a completed trace."""
        with self._lock:
            self._forward_map.pop(trace_id, None)
            self._reverse_map.pop(trace_id, None)
            self._counters.pop(trace_id, None)


# Global in-memory vault singleton
pii_vault = SessionPIIVault()
