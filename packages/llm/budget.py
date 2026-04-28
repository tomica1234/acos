"""Token and context budgeting helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from packages.llm.errors import ContextBudgetExceededError

OutputTokenSetting = int | Literal["auto"]


class TokenBudgetPolicy(BaseModel):
    """Runtime policy for prompt and completion budgeting."""

    model_config = ConfigDict(extra="forbid")

    safety_margin_tokens: int = 4096
    minimum_output_tokens: int = 1024
    default_output_tokens: OutputTokenSetting = "auto"
    hard_max_output_tokens: int | None = None
    estimate_chars_per_token: float = 4.0

    @model_validator(mode="after")
    def validate_limits(self) -> "TokenBudgetPolicy":
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens must be >= 0")
        if self.minimum_output_tokens <= 0:
            raise ValueError("minimum_output_tokens must be > 0")
        if self.hard_max_output_tokens is not None and self.hard_max_output_tokens <= 0:
            raise ValueError("hard_max_output_tokens must be > 0 when provided")
        if self.estimate_chars_per_token <= 0:
            raise ValueError("estimate_chars_per_token must be > 0")
        return self


def estimate_tokens(text: str, *, chars_per_token: float = 4.0) -> int:
    """A rough token estimate suitable for routing and budgeting."""
    if not text:
        return 1
    return max(1, math.ceil(len(text) / chars_per_token))


def estimate_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens from an OpenAI-compatible message array."""
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
    return estimate_tokens(serialized)


def resolve_configured_max_output_tokens(
    *candidates: OutputTokenSetting | None,
) -> OutputTokenSetting | None:
    """Return the first configured output limit, preserving ``auto``."""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def compute_max_output_tokens(
    model_max_context_tokens: int,
    estimated_input_tokens: int,
    configured_max_output_tokens: OutputTokenSetting | None,
    safety_margin_tokens: int = 4096,
    minimum_output_tokens: int = 1024,
    hard_max_output_tokens: int | None = None,
) -> int:
    """Resolve an integer completion budget without overflowing model context."""
    remaining_context = (
        model_max_context_tokens - estimated_input_tokens - safety_margin_tokens
    )
    if remaining_context < minimum_output_tokens:
        raise ContextBudgetExceededError(
            "Context exceeds the available budget after reserving safety margin and minimum output",
            required_tokens=estimated_input_tokens,
        )
    configured_limit = remaining_context
    if isinstance(configured_max_output_tokens, int):
        configured_limit = configured_max_output_tokens
    resolved = min(configured_limit, remaining_context)
    if hard_max_output_tokens is not None:
        resolved = min(resolved, hard_max_output_tokens)
    if resolved <= 0:
        raise ContextBudgetExceededError(
            "Resolved output budget is not positive after applying limits",
            required_tokens=estimated_input_tokens,
        )
    if estimated_input_tokens + resolved + safety_margin_tokens > model_max_context_tokens:
        raise ContextBudgetExceededError(
            "Resolved output budget overflows the selected model context window",
            required_tokens=estimated_input_tokens + resolved + safety_margin_tokens,
        )
    return resolved


def truncate_to_budget(
    text: str,
    token_budget: int,
    *,
    chars_per_token: float = 4.0,
) -> str:
    """Trim text conservatively to fit an approximate token budget."""
    max_chars = max(32, math.floor(token_budget * chars_per_token))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "\n...[truncated]"


@dataclass(slots=True)
class ContextBudget:
    """Allocation buckets for context sections."""

    files: int
    diff: int
    memory: int
    logs: int
    request: int


class TokenBudgetManager:
    """Allocate token budgets across a context packet."""

    def __init__(self, policy: TokenBudgetPolicy | None = None) -> None:
        self.policy = policy or TokenBudgetPolicy()

    def allocate(self, total_budget: int) -> ContextBudget:
        files = int(total_budget * 0.4)
        diff = int(total_budget * 0.2)
        memory = int(total_budget * 0.15)
        logs = int(total_budget * 0.1)
        request = total_budget - files - diff - memory - logs
        return ContextBudget(
            files=files,
            diff=diff,
            memory=memory,
            logs=logs,
            request=request,
        )

    def fit_context_budget(
        self,
        *,
        requested_budget: int,
        model_max_context_tokens: int,
        safety_margin_tokens: int | None = None,
        minimum_output_tokens: int | None = None,
    ) -> int:
        reserved_safety = (
            self.policy.safety_margin_tokens
            if safety_margin_tokens is None
            else safety_margin_tokens
        )
        minimum_output = (
            self.policy.minimum_output_tokens
            if minimum_output_tokens is None
            else minimum_output_tokens
        )
        available = model_max_context_tokens - reserved_safety - minimum_output
        if available <= 0:
            raise ContextBudgetExceededError(
                "Model has no usable prompt budget after reserving safety margin and minimum output",
                required_tokens=requested_budget,
            )
        return min(requested_budget, available)

    def assert_context_fits(
        self,
        *,
        context_tokens: int,
        requested_budget: int,
        model_max_context_tokens: int,
        configured_max_output_tokens: OutputTokenSetting | None = None,
    ) -> int:
        effective_budget = self.fit_context_budget(
            requested_budget=requested_budget,
            model_max_context_tokens=model_max_context_tokens,
        )
        if context_tokens > effective_budget:
            raise ContextBudgetExceededError(
                "Context exceeds selected model budget and requires compaction",
                required_tokens=context_tokens,
            )
        compute_max_output_tokens(
            model_max_context_tokens=model_max_context_tokens,
            estimated_input_tokens=context_tokens,
            configured_max_output_tokens=configured_max_output_tokens,
            safety_margin_tokens=self.policy.safety_margin_tokens,
            minimum_output_tokens=self.policy.minimum_output_tokens,
            hard_max_output_tokens=self.policy.hard_max_output_tokens,
        )
        return effective_budget
