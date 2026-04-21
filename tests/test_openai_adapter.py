from __future__ import annotations

import httpx
import pytest
from openai import APITimeoutError, BadRequestError, InternalServerError

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


def test_openai_adapter_allows_empty_api_key(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "")

    adapter = OpenAICompatibleAdapter(_provider(), _model())

    assert adapter.client.api_key == ""
    assert adapter.client.auth_headers == {}
    assert adapter.client.max_retries == 0


def test_openai_adapter_defaults_missing_api_key_to_empty(monkeypatch) -> None:
    monkeypatch.delenv("TEST_API_KEY", raising=False)

    adapter = OpenAICompatibleAdapter(_provider(), _model())

    assert adapter.client.api_key == ""
    assert adapter.client.auth_headers == {}


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


def test_openai_adapter_normalizes_nested_usage(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret")
    adapter = OpenAICompatibleAdapter(_provider(), _model())

    class FakeUsage:
        def model_dump(self):
            return {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "completion_tokens_details": None,
                "prompt_tokens_details": {"cached_tokens": 3},
            }

    class FakeMessage:
        content = "{\"ok\": true}"
        tool_calls = None

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

        def model_dump(self):
            return {"mock": True}

    adapter.client.chat.completions.create = lambda **kwargs: FakeResponse()

    result = adapter.generate(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        temperature=0.0,
        top_p=None,
        max_tokens=128,
    )

    assert result.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }


def test_openai_adapter_surfaces_provider_status(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret")
    adapter = OpenAICompatibleAdapter(_provider(), _model())
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response = httpx.Response(502, request=request)

    def raise_server_error(**kwargs):
        raise InternalServerError("Error code: 502", response=response, body=None)

    adapter.client.chat.completions.create = raise_server_error

    with pytest.raises(AdapterError) as exc:
        adapter.generate(
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            temperature=0.0,
            top_p=None,
            max_tokens=128,
        )

    assert exc.value.code == "provider_error"
    assert "Provider provider returned HTTP 502 for model provider/model" in str(exc.value)
