"""LLM client responsible for invoking routed model adapters."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from packages.llm.errors import AdapterError
from packages.llm.registry import ModelRegistry
from packages.llm.routing import ModelRouter, RoutingContext
from packages.llm.tool_schema import build_tool_manifest
from packages.schemas.models import (
    ModelCallRecord,
    ModelCallStatus,
    ModelResult,
    ModelSelection,
)


def _hash_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return sha256(text.encode("utf-8")).hexdigest()


class LLMClient:
    """Run routed model calls and return normalized results."""

    def __init__(self, registry: ModelRegistry, router: ModelRouter) -> None:
        self.registry = registry
        self.router = router
        self._adapters: dict[str, object] = {}

    def generate(
        self,
        routing_context: RoutingContext,
        messages: list[dict[str, Any]],
        allowed_tools: list[str],
        response_schema: dict[str, Any] | None,
    ) -> tuple[ModelResult, ModelSelection, ModelCallRecord]:
        selection = self.router.select_model(routing_context)
        adapter = self._get_adapter(selection.model_id)
        tool_payload = build_tool_manifest(allowed_tools) if allowed_tools else None
        try:
            result = adapter.generate(
                messages=messages,
                tools=tool_payload,
                temperature=selection.temperature,
                top_p=selection.top_p,
                max_tokens=selection.max_output_tokens,
                response_schema=response_schema,
                metadata={
                    "role": routing_context.role,
                    "model_key": selection.model_key,
                    "model_name": self.registry.get_model(selection.model_key).model,
                    "provider_name": selection.provider_key,
                },
            )
        except AdapterError:
            raise
        status = ModelCallStatus.SUCCESS
        if selection.reason.value == "fallback":
            status = ModelCallStatus.FALLBACK_USED
        elif selection.reason.value == "escalation":
            status = ModelCallStatus.ESCALATED
        record = ModelCallRecord(
            role=routing_context.role,
            model_key=selection.model_key,
            provider_key=selection.provider_key,
            status=status,
            input_hash=_hash_payload(messages),
            output_hash=_hash_payload(result.model_dump()),
            prompt_tokens_estimate=sum(len(str(item)) for item in messages) // 4,
            completion_tokens_estimate=(
                result.usage.get("completion_tokens", len(result.content) // 4)
                if result.usage is not None
                else len(result.content) // 4
            ),
            total_tokens_estimate=(
                result.usage.get(
                    "total_tokens",
                    sum(len(str(item)) for item in messages) // 4 + len(result.content) // 4,
                )
                if result.usage is not None
                else sum(len(str(item)) for item in messages) // 4 + len(result.content) // 4
            ),
        )
        return result, selection, record

    def _get_adapter(self, model_id: str) -> Any:
        if model_id not in self._adapters:
            self._adapters[model_id] = self.registry.build_adapter(model_id)
        return self._adapters[model_id]
