import pytest
import yaml

from packages.llm.adapters.mock import MockAdapter
from packages.llm.errors import (
    ConfigValidationError,
    UnknownModelError,
    UnknownProviderError,
    UnknownRoleError,
)
from packages.llm.registry import ModelRegistry
from packages.orchestrator.policy import PolicyEngine

from tests.conftest import config_dir, load_registry


def _write_configs(tmp_path, providers: dict, agents: dict, routing: dict):
    provider_path = tmp_path / "providers.yaml"
    agents_path = tmp_path / "agents.yaml"
    routing_path = tmp_path / "routing.yaml"
    provider_path.write_text(yaml.safe_dump(providers), encoding="utf-8")
    agents_path.write_text(yaml.safe_dump(agents), encoding="utf-8")
    routing_path.write_text(yaml.safe_dump(routing), encoding="utf-8")
    return provider_path, agents_path, routing_path


def _base_provider_config() -> dict:
    return {
        "providers": {
            "local_qwen": {
                "type": "openai_compatible",
                "base_url": "http://localhost:8000/v1",
                "api_key_env": "KEY",
                "supports_tools": True,
                "supports_json_mode": False,
                "supports_streaming": False,
                "max_context_tokens": 262144,
                "default_max_output_tokens": 32768,
            },
            "mock_provider": {
                "type": "mock",
                "base_url": "mock://local",
                "api_key_env": "MOCK_KEY",
                "supports_tools": True,
                "supports_json_mode": True,
                "supports_streaming": False,
                "max_context_tokens": 65536,
                "default_max_output_tokens": 8192,
            },
        },
        "models": {
            "qwen_35b": {
                "provider": "local_qwen",
                "model": "qwen/test",
                "display_name": "Qwen",
                "max_context_tokens": 262144,
                "max_output_tokens": 32768,
                "supports_tool_calling": True,
                "supports_structured_output": False,
            },
            "mock_structured": {
                "provider": "mock_provider",
                "model": "mock/structured",
                "display_name": "Mock Structured",
                "max_context_tokens": 65536,
                "max_output_tokens": 8192,
                "supports_tool_calling": True,
                "supports_structured_output": True,
            },
            "mock_small": {
                "provider": "mock_provider",
                "model": "mock/small",
                "display_name": "Mock Small",
                "max_context_tokens": 4096,
                "max_output_tokens": 2048,
                "supports_tool_calling": False,
                "supports_structured_output": True,
            },
        },
    }


def _base_agents_config() -> dict:
    return {
        "agents": {
            "implementer": {
                "primary_model": "qwen_35b",
                "fallback_models": ["mock_structured"],
                "temperature": 0.1,
                "top_p": 0.8,
                "max_output_tokens": 1024,
                "context_budget_tokens": 4096,
                "allow_tools": True,
                "allowed_tools": ["repo_server.apply_patch"],
                "require_json_schema": True,
                "escalation_policy": {},
                "output_schema": "ImplementationResult",
            }
        }
    }


def _base_routing_config() -> dict:
    return {
        "routing": {
            "default_strategy": "role_primary",
            "escalation": {
                "implementer": {
                    "escalate_when": {"repeated_failures_gte": 2},
                    "escalated_model": "qwen_35b",
                }
            },
            "fallback": {"on_errors": ["timeout", "invalid_json"]},
            "capability_requirements": {
                "roles_requiring_tools": ["implementer"],
                "roles_requiring_strict_json": ["implementer"],
            },
        }
    }


def test_model_registry_loads_configs() -> None:
    registry = load_registry()

    provider = registry.get_provider("local_qwen")
    model = registry.get_model("qwen_35b")
    agent = registry.get_agent("pm")

    assert provider.base_url == "http://localhost:8000/v1"
    assert model.provider == "local_qwen"
    assert agent.primary_model == "qwen_35b"

    with pytest.raises(UnknownProviderError):
        registry.get_provider("missing")

    with pytest.raises(UnknownModelError):
        registry.get_model("missing")

    with pytest.raises(UnknownRoleError):
        registry.get_agent("missing")

    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    assert registry.validate(policy=policy) == []


def test_model_registry_rejects_missing_provider_reference(tmp_path) -> None:
    providers = _base_provider_config()
    providers["models"]["qwen_35b"]["provider"] = "missing_provider"
    provider_path, agents_path, routing_path = _write_configs(
        tmp_path,
        providers=providers,
        agents=_base_agents_config(),
        routing=_base_routing_config(),
    )

    with pytest.raises(ConfigValidationError) as exc:
        ModelRegistry.from_paths(provider_path, agents_path, routing_path)

    assert "missing_provider" in str(exc.value)


def test_model_registry_rejects_missing_fallback_model(tmp_path) -> None:
    agents = _base_agents_config()
    agents["agents"]["implementer"]["fallback_models"] = ["missing_model"]
    provider_path, agents_path, routing_path = _write_configs(
        tmp_path,
        providers=_base_provider_config(),
        agents=agents,
        routing=_base_routing_config(),
    )

    with pytest.raises(ConfigValidationError) as exc:
        ModelRegistry.from_paths(provider_path, agents_path, routing_path)

    assert "Agent implementer references unknown model missing_model" in str(exc.value)


def test_model_registry_rejects_tool_incompatible_model_for_tool_role(tmp_path) -> None:
    agents = _base_agents_config()
    agents["agents"]["implementer"]["primary_model"] = "mock_small"
    provider_path, agents_path, routing_path = _write_configs(
        tmp_path,
        providers=_base_provider_config(),
        agents=agents,
        routing=_base_routing_config(),
    )

    with pytest.raises(ConfigValidationError) as exc:
        ModelRegistry.from_paths(provider_path, agents_path, routing_path)

    assert "requires tool calling" in str(exc.value)


def test_model_registry_rejects_context_budget_that_exceeds_any_referenced_model(tmp_path) -> None:
    agents = _base_agents_config()
    agents["agents"]["implementer"]["fallback_models"] = ["mock_small"]
    agents["agents"]["implementer"]["context_budget_tokens"] = 8192
    provider_path, agents_path, routing_path = _write_configs(
        tmp_path,
        providers=_base_provider_config(),
        agents=agents,
        routing=_base_routing_config(),
    )

    with pytest.raises(ConfigValidationError) as exc:
        ModelRegistry.from_paths(provider_path, agents_path, routing_path)

    assert "context budget exceeds model mock_small max_context_tokens" in str(exc.value)


def test_mock_adapter_can_generate_without_external_api() -> None:
    adapter = MockAdapter(
        scenario={
            "summarizer": [
                {
                    "content": "{\"summary\":\"ok\"}",
                    "tool_calls": [
                        {"name": "memory_server.write_memory", "arguments": {"key": "value"}}
                    ],
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    result = adapter.generate(
        messages=[{"role": "user", "content": "summarize"}],
        tools=[{"type": "function", "function": {"name": "memory_server.write_memory"}}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        metadata={"role": "summarizer", "model_name": "mock/model", "provider_name": "mock"},
    )

    assert result.content == "{\"summary\":\"ok\"}"
    assert result.tool_calls[0]["name"] == "memory_server.write_memory"
    assert result.model == "mock/model"
    assert result.provider == "mock"
