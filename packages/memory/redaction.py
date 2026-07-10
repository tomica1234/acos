"""Secret redaction helpers."""

from __future__ import annotations

import re
from typing import Any

REDACTION_RULES: list[tuple[re.Pattern[str], str | None]] = [
    (
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        re.compile(
            r"-----BEGIN OPENSSH PRIVATE KEY-----.*?-----END OPENSSH PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"), "[REDACTED]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[REDACTED]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._-]{10,}", re.IGNORECASE), "Bearer [REDACTED]"),
    (
        re.compile(
            r"((?:api[_-]?key|secret|password|passwd|token|access[_-]?key|secret[_-]?key|aws_access_key_id|aws_secret_access_key|session[_-]?token|private[_-]?key)\s*[:=]\s*)([^\s\"']+)",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        re.compile(
            r"((?:api[_-]?key|secret|password|passwd|token|access[_-]?key|secret[_-]?key|aws_access_key_id|aws_secret_access_key|session[_-]?token|private[_-]?key)\s*[:=]\s*[\"'])(.*?)([\"'])",
            re.IGNORECASE,
        ),
        None,
    ),
]


def redact_text(text: str) -> str:
    """Replace secret-like substrings with a redaction marker."""

    redacted = text
    for pattern, replacement in REDACTION_RULES:
        if replacement is None:
            if pattern.groups >= 3:
                redacted = pattern.sub(
                    lambda match: f"{match.group(1)}[REDACTED]{match.group(3)}",
                    redacted,
                )
            else:
                redacted = pattern.sub(
                    lambda match: f"{match.group(1)}[REDACTED]",
                    redacted,
                )
        else:
            redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_value(value: Any) -> Any:
    """Recursively redact strings inside nested structures."""

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(key): redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    return value
