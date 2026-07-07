"""OpenAI-compatible adapter implementation."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from packages.llm.errors import AdapterError
from packages.memory.redaction import redact_text
from packages.schemas.models import ModelConfig, ModelProviderConfig, ModelResult


class OpenAICompatibleAdapter:
    """Adapter for OpenAI-compatible chat completion APIs."""

    PROMPT_ONLY_STRUCTURED_ROLES = {"implementer", "fixer"}

    def __init__(self, provider: ModelProviderConfig, model: ModelConfig) -> None:
        self.provider = provider
        self.model = model
        api_key = self._resolve_api_key(provider)
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
        if self.provider.extra_body:
            kwargs["extra_body"] = self.provider.extra_body
        used_json_schema = False
        if top_p is not None:
            kwargs["top_p"] = top_p
        if tools:
            kwargs["tools"] = tools
        role = str((metadata or {}).get("role", ""))
        use_provider_json_mode = (
            self.provider.supports_json_mode
            and role not in self.PROMPT_ONLY_STRUCTURED_ROLES
        )
        if response_schema:
            if self.provider.supports_structured_output and self.model.supports_structured_output:
                used_json_schema = True
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self._response_schema_name(response_schema),
                        "strict": True,
                        "schema": response_schema,
                    },
                }
            elif use_provider_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if (
                used_json_schema
                and self.provider.supports_json_mode
                and self._should_downgrade_structured_output(exc)
            ):
                retry_kwargs = dict(kwargs)
                retry_kwargs["response_format"] = {"type": "json_object"}
                try:
                    response = self.client.chat.completions.create(**retry_kwargs)
                except Exception as retry_exc:  # pragma: no cover - network/provider failure
                    raise AdapterError(
                        redact_text(str(retry_exc)),
                        code=self._classify_error(retry_exc),
                    ) from retry_exc
            else:
                raise AdapterError(
                    redact_text(str(exc)),
                    code=self._classify_error(exc),
                ) from exc
        except Exception as exc:  # pragma: no cover - network/provider failure
            raise AdapterError(
                redact_text(str(exc)),
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
                    "type": getattr(item, "type", "function"),
                    "name": item.function.name,
                    "arguments": self._coerce_arguments(item.function.arguments),
                }
                for item in choice.message.tool_calls
            ]
        if response_schema and use_provider_json_mode and content:
            try:
                parsed_content = self._extract_json_object(content)
                content = json.dumps(parsed_content, ensure_ascii=False)
            except (json.JSONDecodeError, ValueError) as exc:
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
    def _response_schema_name(response_schema: dict[str, Any]) -> str:
        raw_name = str(response_schema.get("title") or "structured_output")
        normalized = re.sub(r"[^A-Za-z0-9_-]", "_", raw_name).strip("_")
        if not normalized:
            normalized = "structured_output"
        return normalized[:64]

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any]:
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
        decoder = json.JSONDecoder()
        search_from = 0
        while True:
            start = cleaned.find("{", search_from)
            if start == -1:
                break
            try:
                parsed, _end = decoder.raw_decode(cleaned[start:])
            except json.JSONDecodeError:
                search_from = start + 1
                continue
            if isinstance(parsed, dict):
                return parsed
            search_from = start + 1
        raise ValueError("No JSON object found in structured response")

    @staticmethod
    def _should_downgrade_structured_output(exc: BadRequestError) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "unexpected empty grammar stack",
                "failed to initialize samplers",
                "grammar",
                "json_schema",
                "response_format",
            )
        )

    @staticmethod
    def _normalize_usage(usage: Any) -> dict[str, int] | None:
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            payload = usage.model_dump()
        elif isinstance(usage, dict):
            payload = usage
        else:
            payload = {}
        normalized: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = payload.get(key, getattr(usage, key, None))
            if isinstance(value, int):
                normalized[key] = value
        return normalized or None

    @staticmethod
    def _resolve_api_key(provider: ModelProviderConfig) -> str:
        env_name = provider.api_key_env.strip()
        if env_name:
            value = os.environ.get(env_name)
            if value:
                return value
        if provider.default_api_key is not None:
            return provider.default_api_key
        if provider.allow_empty_api_key:
            return "EMPTY"
        return "missing-api-key"

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        if isinstance(exc, (APITimeoutError, APIConnectionError)):
            return "timeout"
        if isinstance(exc, RateLimitError):
            return "rate_limit"
        if isinstance(exc, APIStatusError):
            if getattr(exc, "status_code", None) == 429:
                return "rate_limit"
            if getattr(exc, "status_code", None) == 408:
                return "timeout"
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
