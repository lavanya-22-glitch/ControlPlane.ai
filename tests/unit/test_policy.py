import pytest
from controlplane.policy.loader import load_policies_from_yaml
from controlplane.policy.registry import PolicyRegistry


def test_load_policies_from_yaml(temp_policy_file):
    policies = load_policies_from_yaml(temp_policy_file)
    assert "customer-support" in policies
    assert "internal-kb-rag" in policies
    assert "action-agent" in policies

    cs = policies["customer-support"]
    assert cs.mode == "chatbot"
    assert cs.pre_execution.block_prompt_injection is True
    assert cs.policy_hash is not None


def test_policy_registry_fallback(temp_policy_file):
    registry = PolicyRegistry(temp_policy_file)
    p = registry.get_policy("non-existent-app-id")
    assert p.version.startswith("default")

    agent_policy = registry.get_policy("action-agent")
    assert agent_policy.mode == "agent"
    assert "execute_command" in agent_policy.tool_guards
