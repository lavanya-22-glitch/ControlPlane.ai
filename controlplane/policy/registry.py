import logging
from pathlib import Path
from typing import Dict, Optional

from controlplane.policy.loader import load_policies_from_yaml
from controlplane.policy.models import PolicyDefinition

logger = logging.getLogger("controlplane.policy")

DEFAULT_FALLBACK_POLICY = PolicyDefinition(
    version="default-1.0.0",
    mode="chatbot",
    fail_mode="fail_closed",
    streaming_mode="buffered",
    policy_hash="default",
)


class PolicyRegistry:
    """Thread-safe in-memory registry of loaded policy rules."""

    def __init__(self, config_path: Optional[str | Path] = None):
        self._policies: Dict[str, PolicyDefinition] = {}
        self._config_path: Optional[Path] = Path(config_path) if config_path else None
        if self._config_path and self._config_path.exists():
            self.reload()

    def reload(self) -> None:
        """Reload policies from the configured YAML path."""
        if not self._config_path:
            logger.warning("No config path specified for PolicyRegistry reload.")
            return

        try:
            self._policies = load_policies_from_yaml(self._config_path)
            logger.info("Loaded %d policies from %s", len(self._policies), self._config_path)
        except Exception as e:
            logger.error("Failed to load policies from %s: %s", self._config_path, e)
            if not self._policies:
                self._policies = {"default": DEFAULT_FALLBACK_POLICY}

    def get_policy(self, app_id: Optional[str]) -> PolicyDefinition:
        """Retrieve policy by App ID or return default fallback."""
        if not app_id or app_id not in self._policies:
            logger.debug("App-ID '%s' not found in registry. Using fallback policy.", app_id)
            return self._policies.get("default", DEFAULT_FALLBACK_POLICY)
        return self._policies[app_id]

    def list_policies(self) -> Dict[str, PolicyDefinition]:
        """Return all active policies in the registry."""
        return dict(self._policies)


# Global registry singleton instance
policy_registry = PolicyRegistry()
