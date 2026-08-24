from typing import Any, List, Optional, Tuple
from controlplane.policy.models import ParameterRule


def check_parameter_bounds(rule: ParameterRule, value: Any) -> Tuple[bool, Optional[str]]:
    """Verify numeric boundaries and categorical allowlists for a parameter value."""
    # 1. Numeric Range Bounds
    if rule.type in ("int", "float", "number") and isinstance(value, (int, float)):
        if rule.min_value is not None and value < rule.min_value:
            return False, f"Parameter '{rule.field}' value {value} is below minimum allowed {rule.min_value}"
        if rule.max_value is not None and value > rule.max_value:
            return False, f"Parameter '{rule.field}' value {value} exceeds maximum allowed {rule.max_value}"

    # 2. Categorical Allowlist
    if rule.allowed_values is not None:
        if value not in rule.allowed_values:
            return False, f"Parameter '{rule.field}' value '{value}' is not in allowed set: {rule.allowed_values}"

    return True, None
