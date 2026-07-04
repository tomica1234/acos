from __future__ import annotations

import json

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
        supports_structured_output=False,
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


def _structured_provider() -> ModelProviderConfig:
    provider = _provider()
    provider.supports_json_mode = True
    provider.supports_structured_output = True
    return provider


def _structured_model() -> ModelConfig:
    model = _model()
    model.supports_structured_output = True
    return model


def test_openai_adapter_classifies_timeout(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret")
    adapter = OpenAICompatibleAdapter(_provider(), _model())
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")

    assert adapter.client.max_retries == 0

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


def test_openai_adapter_normalizes_nested_usage_payload(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret")
    adapter = OpenAICompatibleAdapter(_provider(), _model())

    class FakeFunction:
        name = "repo_server.read_file"
        arguments = "{\"path\":\"README.md\"}"

    class FakeToolCall:
        id = "tool_1"
        function = FakeFunction()

    class FakeMessage:
        content = "{\"summary\":\"ok\"}"
        tool_calls = [FakeToolCall()]

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeUsage:
        prompt_tokens = 123
        completion_tokens = 45
        total_tokens = 168

        def model_dump(self):
            return {
                "prompt_tokens": 123,
                "completion_tokens": 45,
                "total_tokens": 168,
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": None},
                "completion_tokens_details": None,
            }

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

        def model_dump(self):
            return {"choices": [{}], "usage": self.usage.model_dump()}

    adapter.client.chat.completions.create = lambda **kwargs: FakeResponse()

    result = adapter.generate(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "repo_server.read_file"}}],
        temperature=0.0,
        top_p=None,
        max_tokens=128,
    )

    assert result.usage == {
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "total_tokens": 168,
    }
    assert result.tool_calls[0]["arguments"] == {"path": "README.md"}


def test_openai_adapter_uses_json_schema_response_format_when_supported(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret")
    adapter = OpenAICompatibleAdapter(_structured_provider(), _structured_model())
    captured_kwargs: dict[str, object] = {}

    class FakeMessage:
        content = "{\"title\":\"ACOS\",\"problem_statement\":\"Need feature\"}"
        tool_calls = []

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeResponse:
        choices = [FakeChoice()]
        usage = None

        def model_dump(self):
            return {"choices": [{}]}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeResponse()

    adapter.client.chat.completions.create = fake_create

    adapter.generate(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        temperature=0.0,
        top_p=None,
        max_tokens=128,
        response_schema={"title": "PRD", "type": "object", "properties": {}},
    )

    assert captured_kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "PRD",
            "strict": True,
            "schema": {"title": "PRD", "type": "object", "properties": {}},
        },
    }


def test_openai_adapter_uses_json_object_when_only_json_mode_is_supported(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret")
    provider = _provider()
    provider.supports_json_mode = True
    adapter = OpenAICompatibleAdapter(provider, _model())
    captured_kwargs: dict[str, object] = {}

    class FakeMessage:
        content = "{\"summary\":\"ok\"}"
        tool_calls = []

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeResponse:
        choices = [FakeChoice()]
        usage = None

        def model_dump(self):
            return {"choices": [{}]}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeResponse()

    adapter.client.chat.completions.create = fake_create

    adapter.generate(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        temperature=0.0,
        top_p=None,
        max_tokens=128,
        response_schema={"title": "SummaryResult", "type": "object", "properties": {}},
    )

    assert captured_kwargs["response_format"] == {"type": "json_object"}


def test_openai_adapter_downgrades_json_schema_to_json_object_on_grammar_error(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret")
    adapter = OpenAICompatibleAdapter(_structured_provider(), _structured_model())
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response = httpx.Response(400, request=request)
    captured_formats: list[object] = []

    class FakeMessage:
        content = "{\"title\":\"ACOS\",\"problem_statement\":\"Need feature\"}"
        tool_calls = []

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeResponse:
        choices = [FakeChoice()]
        usage = None

        def model_dump(self):
            return {"choices": [{}]}

    def fake_create(**kwargs):
        captured_formats.append(kwargs.get("response_format"))
        if len(captured_formats) == 1:
            raise BadRequestError(
                "Failed to initialize samplers: Unexpected empty grammar stack after accepting piece: <think> (248068)",
                response=response,
                body={},
            )
        return FakeResponse()

    adapter.client.chat.completions.create = fake_create

    adapter.generate(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        temperature=0.0,
        top_p=None,
        max_tokens=128,
        response_schema={"title": "PRD", "type": "object", "properties": {}},
    )

    assert captured_formats == [
        {
            "type": "json_schema",
            "json_schema": {
                "name": "PRD",
                "strict": True,
                "schema": {"title": "PRD", "type": "object", "properties": {}},
            },
        },
        {"type": "json_object"},
    ]


def test_openai_adapter_extracts_json_object_from_think_wrapped_content(monkeypatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret")
    provider = _provider()
    provider.supports_json_mode = True
    adapter = OpenAICompatibleAdapter(provider, _model())

    class FakeMessage:
        content = (
            "<think>internal reasoning</think>\n"
            "{\"title\":\"ACOS\",\"problem_statement\":\"Need feature\"}\n"
            "Extra text that should be ignored."
        )
        tool_calls = []

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeResponse:
        choices = [FakeChoice()]
        usage = None

        def model_dump(self):
            return {"choices": [{}]}

    adapter.client.chat.completions.create = lambda **kwargs: FakeResponse()

    result = adapter.generate(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        temperature=0.0,
        top_p=None,
        max_tokens=128,
        response_schema={"title": "PRD", "type": "object", "properties": {}},
    )

    assert json.loads(result.content) == {
        "title": "ACOS",
        "problem_statement": "Need feature",
    }
