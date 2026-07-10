"""Deterministic memory summarization helpers."""

from __future__ import annotations

from packages.schemas.agent_outputs import SummaryResult


def summarize_memory(entries: list[str], max_items: int = 5) -> SummaryResult:
    """Create a short SummaryResult from plain memory entries."""
    trimmed = [item.strip() for item in entries if item.strip()][:max_items]
    summary = "; ".join(trimmed[:3]) if trimmed else "No memory entries captured."
    return SummaryResult(summary=summary, memory_entries=trimmed)

