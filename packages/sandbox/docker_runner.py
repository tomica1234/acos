"""Docker sandbox runner skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class DockerRunRequest:
    image: str
    command: Sequence[str]
    workspace: str
    timeout_seconds: int = 600
    max_memory: str = "8g"
    max_cpus: int = 4


@dataclass(slots=True)
class DockerRunResult:
    success: bool
    command: list[str]
    output: str
    exit_code: int


class DockerSandboxRunner:
    """Skeleton for future isolated execution."""

    def run(self, request: DockerRunRequest) -> DockerRunResult:
        return DockerRunResult(
            success=False,
            command=list(request.command),
            output="Docker sandbox runner is not implemented in the MVP.",
            exit_code=127,
        )

