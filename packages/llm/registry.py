"""Configuration-backed registry for providers, models, and roles."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

import yaml

from packages.llm.adapters.mock import MockAdapter
from packages.llm.adapters.openai_compatible import OpenAICompatibleAdapter
from packages.llm.errors import (
    ConfigValidationError,
    UnknownModelError,
    UnknownProviderError,
    UnknownRoleError,
)
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.models import (
    AgentModelConfig,
    ModelConfig,
    ModelProviderConfig,
    ModelRoutingConfig,
    ProviderType,
)

AdapterFactory = Callable[[ModelProviderConfig, ModelConfig], object]
_ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)(?::-([^}]*))?\}$")


class ModelRegistry:
    """Load and expose provider/model/role configuration."""

    def __init__(
        self,
        providers: dict[str, ModelProviderConfig],
        models: dict[str, ModelConfig],
        agents: dict[str, AgentModelConfig],
        routing: ModelRoutingConfig,
    ) -> None:
        self.providers = providers
        self.models = models
        self.agents = agents
        self.routing = routing
        self._adapter_factories: dict[ProviderType, AdapterFactory] = {
            ProviderType.OPENAI_COMPATIBLE: lambda provider, model: OpenAICompatibleAdapter(
                provider, model
            ),
            ProviderType.MOCK: lambda provider, model: MockAdapter(),
        }

    @classmethod
    def load_from_paths(
        cls,
        provider_path: str | Path,
        agents_path: str | Path,
        routing_path: str | Path,
    ) -> "ModelRegistry":
        cls._load_env_file(Path(provider_path).resolve().parents[1] / ".env")
        provider_data = cls._load_yaml(provider_path)
        providers: dict[str, ModelProviderConfig] = {}
        for name, payload in provider_data.get("providers", {}).items():
            providers[name] = ModelProviderConfig(name=name, **payload)
        models: dict[str, ModelConfig] = {}
        for model_id, payload in provider_data.get("models", {}).items():
            models[model_id] = ModelConfig(model_id=model_id, **payload)
        agents_data = cls._load_yaml(agents_path)
        agents: dict[str, AgentModelConfig] = {}
        for role, payload in agents_data.get("agents", {}).items():
            agents[role] = AgentModelConfig(role=role, **payload)
        routing_data = cls._load_yaml(routing_path)
        routing = ModelRoutingConfig(**routing_data["routing"])
        return cls(providers=providers, models=models, agents=agents, routing=routing)

    @classmethod
    def from_paths(
        cls,
        provider_path: str | Path,
        agents_path: str | Path,
        routing_path: str | Path,
    ) -> "ModelRegistry":
        registry = cls.load_from_paths(
            provider_path=provider_path,
            agents_path=agents_path,
            routing_path=routing_path,
        )
        registry.validate_or_raise()
        return registry

    @staticmethod
    def _load_yaml(path: str | Path) -> dict:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return ModelRegistry._expand_env_placeholders(data)

    @staticmethod
    def _expand_env_placeholders(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: ModelRegistry._expand_env_placeholders(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [ModelRegistry._expand_env_placeholders(item) for item in value]
        if isinstance(value, str):
            match = _ENV_PATTERN.match(value)
            if match is None:
                return value
            env_name, default = match.groups()
            resolved = os.environ.get(env_name)
            if resolved is not None and resolved != "":
                return resolved
            if default is not None:
                return default
        return value

    @staticmethod
    def _load_env_file(path: Path) -> None:
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = raw_value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {'"', "'"}
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)

    def register_adapter_factory(
        self, provider_type: ProviderType, factory: AdapterFactory
    ) -> None:
        self._adapter_factories[provider_type] = factory

    def get_provider(self, name: str) -> ModelProviderConfig:
        try:
            return self.providers[name]
        except KeyError as exc:
            raise UnknownProviderError(name) from exc

    def get_model(self, model_id: str) -> ModelConfig:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise UnknownModelError(model_id) from exc

    def get_agent(self, role: str) -> AgentModelConfig:
        try:
            return self.agents[role]
        except KeyError as exc:
            raise UnknownRoleError(role) from exc

    def build_adapter(self, model_id: str) -> object:
        model = self.get_model(model_id)
        provider = self.get_provider(model.provider)
        factory = self._adapter_factories[provider.type]
        return factory(provider, model)

    def list_models(self) -> list[ModelConfig]:
        return [self.models[key] for key in sorted(self.models)]

    def list_agents(self) -> list[AgentModelConfig]:
        return [self.agents[key] for key in sorted(self.agents)]

    def resolve_primary_model(self, role: str) -> ModelConfig:
        agent = self.get_agent(role)
        return self.get_model(agent.primary_model)

    def resolve_fallback_models(self, role: str) -> list[ModelConfig]:
        agent = self.get_agent(role)
        return [self.get_model(model_key) for model_key in agent.fallback_models]

    def validate(self, policy: PolicyEngine | None = None) -> list[str]:
        errors: list[str] = []
        if not self.providers:
            errors.append("No providers are configured")
        if not self.models:
            errors.append("No models are configured")
        if not self.agents:
            errors.append("No agents are configured")
        for model_key, model in self.models.items():
            if model.provider not in self.providers:
                errors.append(f"Model {model_key} references unknown provider {model.provider}")
                continue
            provider = self.providers[model.provider]
            if (
                provider.max_context_tokens is not None
                and model.max_context_tokens > provider.max_context_tokens
            ):
                errors.append(
                    f"Model {model_key} max_context_tokens exceeds provider {provider.name}"
                )
            if (
                provider.default_max_output_tokens is not None
                and model.max_output_tokens > provider.default_max_output_tokens
            ):
                errors.append(
                    f"Model {model_key} max_output_tokens exceeds provider {provider.name}"
                )
            if model.supports_tool_calling and not provider.supports_tools:
                errors.append(
                    f"Model {model_key} requires tool calling but provider {provider.name} disables tools"
                )
        for role, agent in self.agents.items():
            referenced = [agent.primary_model, *agent.fallback_models]
            for model_key in referenced:
                if model_key not in self.models:
                    errors.append(f"Agent {role} references unknown model {model_key}")
                    continue
                model = self.models[model_key]
                if agent.context_budget_tokens > model.max_context_tokens:
                    errors.append(
                        f"Agent {role} context budget exceeds model {model_key} max_context_tokens"
                    )
                if (
                    model_key == agent.primary_model
                    and agent.max_output_tokens > model.max_output_tokens
                ):
                    errors.append(
                        f"Agent {role} max_output_tokens exceeds model {model_key} max_output_tokens"
                    )
                if (
                    role
                    in self.routing.capability_requirements.roles_requiring_tools
                    and not model.supports_tool_calling
                ):
                    errors.append(
                        f"Agent {role} requires tool calling but model {model_key} does not support it"
                    )
                if (
                    role
                    in self.routing.capability_requirements.roles_requiring_strict_json
                    and not (model.supports_structured_output or model.supports_json_repair)
                ):
                    errors.append(
                        f"Agent {role} requires strict JSON but model {model_key} supports neither structured output nor repair"
                    )
            if policy is not None:
                allowed = set(policy.config.tools.allow_by_role.get(role, []))
                for tool_name in agent.allowed_tools:
                    if tool_name not in allowed:
                        errors.append(
                            f"Agent {role} allows tool {tool_name} that policy does not allow"
                        )
        for role, config in self.routing.escalation.items():
            if role not in self.agents:
                errors.append(f"Routing escalation references unknown role {role}")
                continue
            if config.escalated_model not in self.models:
                errors.append(
                    f"Routing escalation for {role} references unknown model {config.escalated_model}"
                )
                continue
            agent = self.agents[role]
            if config.escalated_model == agent.primary_model:
                errors.append(
                    f"Routing escalation for {role} must use a different model than primary {agent.primary_model}"
                )
        return errors

    def validate_or_raise(self, policy: PolicyEngine | None = None) -> None:
        errors = self.validate(policy=policy)
        if errors:
            raise ConfigValidationError(errors)
