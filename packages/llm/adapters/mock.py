"""Deterministic mock adapter used in tests and demos."""

from __future__ import annotations

import json
from collections import defaultdict
from enum import Enum
from typing import Any, Callable

from packages.schemas.models import ModelResult

ScenarioItem = dict[str, Any] | str | Callable[[dict[str, Any] | None], dict[str, Any] | str]


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return str(value)


class MockAdapter:
    """A configurable adapter that returns pre-baked responses per role."""

    def __init__(
        self,
        scenario: dict[str, list[ScenarioItem] | ScenarioItem] | None = None,
        default_payload: dict[str, Any] | None = None,
    ) -> None:
        self._queues: dict[str, list[ScenarioItem]] = defaultdict(list)
        self.default_payload = default_payload or {"summary": "mock response"}
        if scenario:
            for role, payload in scenario.items():
                if isinstance(payload, list):
                    self._queues[role] = list(payload)
                else:
                    self._queues[role] = [payload]

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
        role = "default"
        model_name = "mock/response"
        provider_name = "mock_provider"
        if metadata is not None:
            role = str(metadata.get("role", role))
            model_name = str(metadata.get("model_name", model_name))
            provider_name = str(metadata.get("provider_name", provider_name))
        payload: ScenarioItem
        queue = self._queues.get(role) or self._queues.get("default")
        if queue:
            payload = queue.pop(0)
        else:
            payload = (
                _synthesize_payload_from_schema(response_schema)
                if response_schema is not None
                else self.default_payload
            )
        if callable(payload):
            payload = payload(metadata)
        if isinstance(payload, str):
            return ModelResult(
                content=payload,
                tool_calls=[],
                raw={"mock": True},
                model=model_name,
                provider=provider_name,
                finish_reason="stop",
            )
        if {"content", "tool_calls", "raw", "finish_reason", "usage"} & set(payload):
            return ModelResult(
                content=str(payload.get("content", "")),
                tool_calls=list(payload.get("tool_calls", [])),
                raw=payload.get("raw", {"mock": True, "message_count": len(messages)}),
                model=str(payload.get("model", model_name)),
                provider=str(payload.get("provider", provider_name)),
                finish_reason=str(payload.get("finish_reason", "stop")),
                usage=payload.get("usage"),
            )
        return ModelResult(
            content=json.dumps(payload, default=_json_default),
            tool_calls=[],
            raw={
                "mock": True,
                "tools_requested": bool(tools),
                "message_count": len(messages),
            },
            model=model_name,
            provider=provider_name,
            finish_reason="stop",
        )


def _synthesize_payload_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {}) if isinstance(schema.get("$defs"), dict) else {}
    value = _schema_value(schema, defs=defs, field_name="")
    if isinstance(value, dict):
        return value
    return {"value": value}


def _schema_value(
    schema: dict[str, Any] | None,
    *,
    defs: dict[str, Any],
    field_name: str,
) -> Any:
    if not isinstance(schema, dict):
        return "mock"
    if "$ref" in schema:
        ref = str(schema["$ref"])
        if ref.startswith("#/$defs/"):
            return _schema_value(defs.get(ref.split("/", 2)[-1]), defs=defs, field_name=field_name)
        return "mock"
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list):
            for option in options:
                if isinstance(option, dict) and option.get("type") == "null":
                    continue
                return _schema_value(option, defs=defs, field_name=field_name)
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        result: dict[str, Any] = {}
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str):
                    result[key] = _schema_value(
                        properties.get(key, {}),
                        defs=defs,
                        field_name=key,
                    )
        return result
    if schema_type == "array":
        return []
    if schema_type == "boolean":
        return False
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0
    if schema_type == "string":
        lowered = field_name.lower()
        if "title" in lowered:
            return "mock title"
        if "problem" in lowered:
            return "mock problem statement"
        if "summary" in lowered:
            return "mock summary"
        if "description" in lowered:
            return "mock description"
        if "path" in lowered:
            return "mock/path"
        return "mock"
    return "mock"
