"""Common adapter interfaces."""

from __future__ import annotations

from typing import Any, Protocol

from packages.schemas.models import ModelResult


class ModelAdapter(Protocol):
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
        ...

