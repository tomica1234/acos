"""Definition of Done verification for autonomous jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packages.schemas.jobs import JobRecord
from packages.schemas.models import ReviewDecision


@dataclass
class CompletionVerification:
    passed: bool
    missing_evidence: list[str] = field(default_factory=list)
    unresolved_findings: list[str] = field(default_factory=list)


class DefinitionOfDoneVerifier:
    """Check whether a job has enough evidence to enter DONE."""

    def verify(self, record: JobRecord) -> CompletionVerification:
        missing: list[str] = []
        unresolved: list[str] = []
        outputs = record.outputs

        task_graph = outputs.get("task_graph")
        tasks = task_graph.get("tasks", []) if isinstance(task_graph, dict) else []
        planned_ids = {str(task.get("id")) for task in tasks if isinstance(task, dict) and task.get("id")}
        completed_ids = set(record.completed_task_ids)
        for task_id in sorted(planned_ids - completed_ids):
            missing.append(f"planned_task_not_done:{task_id}")

        for artifact in self._required_artifacts(outputs):
            if not self._artifact_exists(record, artifact):
                missing.append(f"required_artifact_missing:{artifact}")
        for target in self._target_files(outputs):
            if not self._artifact_exists(record, target):
                missing.append(f"target_file_missing:{target}")

        test_run = outputs.get("test_run")
        if not isinstance(test_run, dict) or test_run.get("success") is not True:
            missing.append("unit_tests_success")
        if "runtime_smoke" in outputs and not self._success_value(outputs.get("runtime_smoke")):
            missing.append("runtime_smoke_success")
        if "acceptance_checks" in outputs and not self._success_value(outputs.get("acceptance_checks")):
            missing.append("acceptance_checks_success")

        review = outputs.get("reviewer") or outputs.get("review")
        if isinstance(review, dict) and review.get("decision") != ReviewDecision.APPROVE.value:
            missing.append("reviewer_approve")
        security = outputs.get("security_reviewer") or outputs.get("security_review")
        if isinstance(security, dict):
            if security.get("decision") != ReviewDecision.APPROVE.value:
                missing.append("security_reviewer_approve")
            for finding in security.get("findings", []) or []:
                if isinstance(finding, dict) and finding.get("severity") in {"high", "critical"}:
                    unresolved.append(str(finding.get("title") or finding.get("description") or "security finding"))

        if not record.audit_events:
            missing.append("audit_evidence")
        if not record.checkpoints:
            missing.append("checkpoint_evidence")
        if not outputs.get("completion_integrity", {"passed": True}).get("passed", True):
            missing.append("completion_integrity_pass")

        return CompletionVerification(
            passed=not missing and not unresolved,
            missing_evidence=missing,
            unresolved_findings=unresolved,
        )

    @staticmethod
    def _success_value(value: Any) -> bool:
        return bool(isinstance(value, dict) and value.get("success") is True)

    @staticmethod
    def _required_artifacts(outputs: dict[str, Any]) -> set[str]:
        return DefinitionOfDoneVerifier._collect_task_paths(outputs, "required_artifacts")

    @staticmethod
    def _target_files(outputs: dict[str, Any]) -> set[str]:
        return DefinitionOfDoneVerifier._collect_task_paths(outputs, "target_files")

    @staticmethod
    def _collect_task_paths(outputs: dict[str, Any], key: str) -> set[str]:
        task_graph = outputs.get("task_graph")
        tasks = task_graph.get("tasks", []) if isinstance(task_graph, dict) else []
        paths: set[str] = set()
        for task in tasks:
            if not isinstance(task, dict):
                continue
            for path in task.get(key, []) or []:
                if isinstance(path, str) and path.strip():
                    paths.add(path)
        return paths

    @staticmethod
    def _artifact_exists(record: JobRecord, relative_path: str) -> bool:
        root = Path(record.spec.workspace_root or record.spec.repo_path)
        return (root / relative_path).exists()
