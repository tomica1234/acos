"""Structured schemas used across ACOS.

The package re-exports common schema classes lazily so importing one schema
module does not pull the whole orchestrator stack into memory.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AgentModelConfig": "packages.schemas.models",
    "ArchitecturePlan": "packages.schemas.agent_outputs",
    "AuditEvent": "packages.schemas.audit",
    "ApprovalChallenge": "packages.schemas.approvals",
    "ApprovalRequest": "packages.schemas.approvals",
    "CheckpointRecord": "packages.schemas.checkpoints",
    "ContextPacket": "packages.schemas.context",
    "FilePatch": "packages.schemas.agent_outputs",
    "Finding": "packages.schemas.agent_outputs",
    "FixResult": "packages.schemas.agent_outputs",
    "ImplementationResult": "packages.schemas.agent_outputs",
    "JobLease": "packages.schemas.runtime",
    "JobRecord": "packages.schemas.jobs",
    "JobSpec": "packages.schemas.jobs",
    "ModelCallRecord": "packages.schemas.models",
    "ModelConfig": "packages.schemas.models",
    "ModelProviderConfig": "packages.schemas.models",
    "ModelResult": "packages.schemas.models",
    "ModelRoutingConfig": "packages.schemas.models",
    "ModelSelection": "packages.schemas.models",
    "PRD": "packages.schemas.agent_outputs",
    "PlannedTask": "packages.schemas.tasks",
    "ReleaseResult": "packages.schemas.agent_outputs",
    "ReviewResult": "packages.schemas.agent_outputs",
    "RuntimeIssue": "packages.schemas.runtime",
    "SecurityReviewResult": "packages.schemas.agent_outputs",
    "SummaryResult": "packages.schemas.agent_outputs",
    "TaskGraph": "packages.schemas.tasks",
    "TaskRecord": "packages.schemas.tasks",
    "TestRunResult": "packages.schemas.agent_outputs",
    "TestWriterResult": "packages.schemas.agent_outputs",
    "WorkerHeartbeat": "packages.schemas.runtime",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
