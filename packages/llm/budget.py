"""Token and context budgeting helpers."""

from __future__ import annotations

from dataclasses import dataclass

from packages.llm.errors import ContextBudgetExceededError


def estimate_tokens(text: str) -> int:
    """A rough token estimate suitable for local routing decisions."""
    return max(1, len(text) // 4)


def truncate_to_budget(text: str, token_budget: int) -> str:
    """Trim text conservatively to fit an approximate token budget."""
    max_chars = max(32, token_budget * 4)
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
        model_max_output_tokens: int,
        overhead_tokens: int = 512,
    ) -> int:
        available = model_max_context_tokens - model_max_output_tokens - overhead_tokens
        if available <= 0:
            raise ContextBudgetExceededError(
                "Model has no usable prompt budget after reserving output tokens",
                required_tokens=requested_budget,
            )
        return min(requested_budget, available)

    def assert_context_fits(
        self,
        *,
        context_tokens: int,
        requested_budget: int,
        model_max_context_tokens: int,
        model_max_output_tokens: int,
    ) -> None:
        effective_budget = self.fit_context_budget(
            requested_budget=requested_budget,
            model_max_context_tokens=model_max_context_tokens,
            model_max_output_tokens=model_max_output_tokens,
        )
        if context_tokens > effective_budget:
            raise ContextBudgetExceededError(
                "Context exceeds selected model budget and requires compaction",
                required_tokens=context_tokens,
            )
