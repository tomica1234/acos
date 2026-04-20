"""Helpers to build model message payloads."""

from __future__ import annotations

from typing import Any

from packages.schemas.context import ContextPacket


def build_messages(system_prompt: str, context_packet: ContextPacket) -> list[dict[str, Any]]:
    """Build a standard message array for a role invocation."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_packet.render_text()},
    ]

