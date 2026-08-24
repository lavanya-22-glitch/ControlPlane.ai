import hashlib
import os
from pathlib import Path
from typing import Dict
import yaml

from controlplane.policy.models import PolicyDefinition, PoliciesFile


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a policy file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def load_policies_from_yaml(file_path: str | Path) -> Dict[str, PolicyDefinition]:
    """Parse YAML policy file and attach policy hash."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Policy configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    parsed_file = PoliciesFile.model_validate(raw_data)
    file_hash = compute_file_hash(path)

    for app_id, policy in parsed_file.policies.items():
        policy.policy_hash = file_hash

    return parsed_file.policies
