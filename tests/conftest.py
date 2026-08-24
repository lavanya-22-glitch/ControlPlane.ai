import pytest
import os
import tempfile
from pathlib import Path
import yaml

from controlplane.policy.registry import PolicyRegistry
from controlplane.telemetry.sink import TelemetrySink


@pytest.fixture
def sample_policy_dict():
    return {
        "policies": {
            "customer-support": {
                "version": "1.0.0",
                "mode": "chatbot",
                "fail_mode": "fail_closed",
                "streaming_mode": "buffered",
                "pre_execution": {
                    "block_prompt_injection": True,
                    "injection_threshold": 0.60,
                    "pii_action": "mask",
                    "masked_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "PERSON"],
                },
                "post_execution": {
                    "toxicity_threshold": 0.70,
                    "bias_subspace_threshold": 0.65,
                    "action_on_violation": "rewrite",
                    "fallback_message": "Safe fallback reply.",
                },
            },
            "internal-kb-rag": {
                "version": "1.0.0",
                "mode": "rag",
                "fail_mode": "fail_open",
                "streaming_mode": "buffered",
                "pre_execution": {
                    "block_prompt_injection": True,
                    "injection_threshold": 0.70,
                    "pii_action": "mask",
                    "masked_entities": ["EMAIL_ADDRESS"],
                },
                "post_execution": {
                    "hallucination_engine": {
                        "enabled": True,
                        "min_grounding_score": 0.70,
                        "action_on_low_grounding": "rewrite",
                        "fallback_message": "Unverified claim fallback.",
                    }
                },
            },
            "action-agent": {
                "version": "1.0.0",
                "mode": "agent",
                "fail_mode": "fail_closed",
                "streaming_mode": "pass_through",
                "tool_guards": {
                    "execute_command": {
                        "parameter_rules": [
                            {"field": "command_type", "type": "string", "allowed_values": ["READ", "QUERY"]},
                            {"field": "timeout_seconds", "type": "int", "min_value": 1, "max_value": 60},
                        ],
                        "action_on_violation": "block",
                    }
                },
            },
        }
    }


@pytest.fixture
def temp_policy_file(sample_policy_dict):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_policy_dict, f)
        temp_path = f.name

    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile("w", suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)
