from __future__ import annotations

import os
from types import SimpleNamespace

from packages.orchestrator.provider_health import ProviderHealthChecker
from packages.schemas.models import ModelProviderConfig, ProviderType
from packages.schemas.runtime import ProviderHealthStatus
from tests.conftest import load_registry


class _FakeClient:
    def __init__(self, model_ids: list[str]) -> None:
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id=item) for item in model_ids])
        )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: {"ok": True, "kwargs": kwargs}
            )
        )


def test_provider_health_reports_models_endpoint_ok(monkeypatch) -> None:
    registry = load_registry()
    checker = ProviderHealthChecker(registry)
    monkeypatch.setattr(checker, "_build_client", lambda provider: _FakeClient(["foo"]))

    health = checker.check_provider("local_qwen")

    assert health.status == ProviderHealthStatus.OK


def test_provider_health_reports_model_exists(monkeypatch) -> None:
    registry = load_registry()
    checker = ProviderHealthChecker(registry)
    model = registry.get_model("qwen_35b")
    monkeypatch.setattr(checker, "_build_client", lambda provider: _FakeClient([model.model, "other"]))

    health = checker.check_model("qwen_35b")

    assert health.status == ProviderHealthStatus.OK


def test_provider_health_accepts_extensionless_reported_model_id(monkeypatch) -> None:
    registry = load_registry()
    checker = ProviderHealthChecker(registry)
    model = registry.get_model("qwen_35b")
    reported_model_id = model.model.removesuffix(".gguf")
    monkeypatch.setattr(
        checker,
        "_build_client",
        lambda provider: _FakeClient([reported_model_id]),
    )

    health = checker.check_model("qwen_35b")

    assert health.status == ProviderHealthStatus.OK
    assert health.model_available is True


def test_provider_health_reports_model_missing(monkeypatch) -> None:
    registry = load_registry()
    checker = ProviderHealthChecker(registry)
    monkeypatch.setattr(checker, "_build_client", lambda provider: _FakeClient(["other-model"]))

    health = checker.check_model("qwen_35b")

    assert health.status == ProviderHealthStatus.MODEL_NOT_FOUND


def test_provider_health_reports_connection_error(monkeypatch) -> None:
    registry = load_registry()
    checker = ProviderHealthChecker(registry)
    monkeypatch.setattr(checker, "_build_client", lambda provider: (_ for _ in ()).throw(ConnectionError("down")))

    health = checker.check_provider("local_qwen")

    assert health.status == ProviderHealthStatus.CONNECTION_ERROR


def test_provider_health_uses_empty_api_key_when_allowed(monkeypatch) -> None:
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    provider = ModelProviderConfig(
        name="local",
        type=ProviderType.OPENAI_COMPATIBLE,
        base_url="http://localhost:8000/v1",
        api_key_env="QWEN_API_KEY",
        allow_empty_api_key=True,
        default_api_key="EMPTY",
    )

    assert ProviderHealthChecker._resolve_api_key(provider) == "EMPTY"


def test_provider_health_does_not_expose_env_api_key_in_status_payload(monkeypatch) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "SUPER_SECRET")
    registry = load_registry()
    checker = ProviderHealthChecker(registry)
    monkeypatch.setattr(checker, "_build_client", lambda provider: _FakeClient(["foo"]))

    payload = checker.check_provider("local_qwen").model_dump_json()

    assert "SUPER_SECRET" not in payload
