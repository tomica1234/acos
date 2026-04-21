"""OpenAI-compatible adapter implementation."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

from packages.llm.errors import AdapterError
from packages.memory.redaction import redact_text
from packages.schemas.models import ModelConfig, ModelProviderConfig, ModelResult


class OpenAICompatibleAdapter:
    """Adapter for OpenAI-compatible chat completion APIs."""

    def __init__(self, provider: ModelProviderConfig, model: ModelConfig) -> None:
        self.provider = provider
        self.model = model
        api_key_env = provider.api_key_env.strip()
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if not api_key and provider.allow_empty_api_key:
            api_key = provider.default_api_key or "EMPTY"
        if not api_key and provider.default_api_key is not None:
            api_key = provider.default_api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url=provider.base_url,
            timeout=provider.timeout_seconds,
            max_retries=0,
            default_headers=provider.default_headers,
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        top_p: float | None,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelResult:
        kwargs: dict[str, Any] = {
            "model": self.model.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if tools:
            kwargs["tools"] = tools
        if response_schema and self.provider.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # pragma: no cover - network/provider failure
            raise AdapterError(
                self._format_error_message(exc),
                code=self._classify_error(exc),
            ) from exc
        choice = response.choices[0]
        content = choice.message.content or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        tool_calls = []
        if getattr(choice.message, "tool_calls", None):
            tool_calls = [
                {
                    "id": item.id,
                    "name": item.function.name,
                    "arguments": self._coerce_arguments(item.function.arguments),
                }
                for item in choice.message.tool_calls
            ]
        if response_schema and self.provider.supports_json_mode and content:
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                raise AdapterError(
                    "Provider returned invalid JSON for a structured response",
                    code="invalid_json",
                ) from exc
        return ModelResult(
            content=content,
            tool_calls=tool_calls,
            raw=response.model_dump(),
            model=self.model.model_id,
            provider=self.provider.name,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=self._normalize_usage(getattr(response, "usage", None)),
        )

    @staticmethod
    def _coerce_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {"raw_arguments": arguments}
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        return {"value": arguments}

    @staticmethod
    def _normalize_usage(usage: Any) -> dict[str, int] | None:
        if usage is None:
            return None
        payload = usage.model_dump() if hasattr(usage, "model_dump") else usage
        if not isinstance(payload, dict):
            return None
        normalized: dict[str, int] = {}
        for key, value in payload.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                normalized[key] = value
        return normalized or None

    def _format_error_message(self, exc: Exception) -> str:
        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            if status_code is None and getattr(exc, "response", None) is not None:
                status_code = exc.response.status_code
            message = (
                f"Provider {self.provider.name} returned HTTP {status_code} "
                f"for model {self.model.model}"
            )
            body = getattr(exc, "body", None)
            if body not in (None, ""):
                serialized = (
                    json.dumps(body, sort_keys=True, default=str)
                    if not isinstance(body, str)
                    else body
                )
                message = f"{message}: {serialized}"
            elif getattr(exc, "response", None) is not None:
                try:
                    response_text = exc.response.text.strip()
                except Exception:  # pragma: no cover - defensive
                    response_text = ""
                if response_text:
                    message = f"{message}: {response_text}"
            return redact_text(message)
        return redact_text(str(exc))

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        if isinstance(exc, AuthenticationError):
            return "auth_error"
        if isinstance(exc, (APITimeoutError, APIConnectionError)):
            return "timeout"
        if isinstance(exc, RateLimitError):
            return "rate_limit"
        if isinstance(exc, NotFoundError):
            return "model_not_found"
        if isinstance(exc, APIStatusError):
            if getattr(exc, "status_code", None) == 429:
                return "rate_limit"
            if getattr(exc, "status_code", None) == 408:
                return "timeout"
            if getattr(exc, "status_code", None) == 401:
                return "auth_error"
            if getattr(exc, "status_code", None) == 404:
                return "model_not_found"
        if isinstance(exc, BadRequestError):
            message = str(exc).lower()
            if "context" in message and any(
                marker in message for marker in ("length", "window", "maximum", "max tokens")
            ):
                return "context_overflow"
            if "tool" in message and any(
                marker in message for marker in ("unsupported", "not support", "does not support")
            ):
                return "tool_call_unsupported"
        return "provider_error"
