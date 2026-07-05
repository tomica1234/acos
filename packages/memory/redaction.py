"""Secret redaction helpers."""

from __future__ import annotations

import re

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{10,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[:=]\s*)([^\s]+)", re.IGNORECASE),
]


def redact_text(text: str) -> str:
    """Replace secret-like substrings with a redaction marker."""
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(r"\1[REDACTED]" if pattern.groups else "[REDACTED]", redacted)
    return redacted

