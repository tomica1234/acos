from __future__ import annotations

import httpx
import pytest
from openai import APITimeoutError, BadRequestError

from packages.llm.adapters.openai_compatible import OpenAICompatibleAdapter
from packages.llm.errors import AdapterError
from packages.schemas.models import ModelConfig, ModelProviderConfig, ProviderType


def _provider() -> ModelProviderConfig:
    return ModelProviderConfig(
        name="provider",
        type=ProviderType.OPENAI_COMPATIBLE,
        base_url="http://localhost:8000/v1",
        api_key_env="TEST_API_KEY",
        timeout_seconds=30,
        supports_tools=True,
        supports_json_mode=False,
    )


def _model() -> ModelConfig:
    return ModelConfig(
        model_id="model_key",
        provider="provider",
        model="provider/model",
        display_name="Model",
        max_context_tokens=32768,
        max_output_tokens=4096,
        supports_tool_calling=True,
        supports_structured_output=False,
    )


def test_openai_adapter_classifies_timeout(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret")
    adapter = OpenAICompatibleAdapter(_provider(), _model())
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")

    def raise_timeout(**kwargs):
        raise APITimeoutError(request=request)

    adapter.client.chat.completions.create = raise_timeout

    with pytest.raises(AdapterError) as exc:
        adapter.generate(
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            temperature=0.0,
            top_p=None,
            max_tokens=128,
        )

    assert exc.value.code == "timeout"
    assert "sk-test-secret" not in str(exc.value)


def test_openai_adapter_classifies_context_overflow(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret")
    adapter = OpenAICompatibleAdapter(_provider(), _model())
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response = httpx.Response(400, request=request)

    def raise_bad_request(**kwargs):
        raise BadRequestError(
            "This model's maximum context length is 8192 tokens.",
            response=response,
            body={},
        )

    adapter.client.chat.completions.create = raise_bad_request

    with pytest.raises(AdapterError) as exc:
        adapter.generate(
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            temperature=0.0,
            top_p=None,
            max_tokens=128,
        )

    assert exc.value.code == "context_overflow"
    assert "sk-test-secret" not in str(exc.value)
