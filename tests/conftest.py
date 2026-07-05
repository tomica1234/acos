from __future__ import annotations

from pathlib import Path

from packages.llm.adapters.mock import MockAdapter
from packages.llm.registry import ModelRegistry
from packages.schemas.models import ProviderType


def config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def load_registry() -> ModelRegistry:
    base = config_dir()
    return ModelRegistry.from_paths(
        provider_path=base / "model_providers.yaml",
        agents_path=base / "agents.yaml",
        routing_path=base / "model_routing.yaml",
    )


def attach_mock_adapter(registry: ModelRegistry, scenario: dict) -> MockAdapter:
    adapter = MockAdapter(scenario=scenario)
    registry.register_adapter_factory(ProviderType.OPENAI_COMPATIBLE, lambda provider, model: adapter)
    registry.register_adapter_factory(ProviderType.MOCK, lambda provider, model: adapter)
    return adapter

