"""Definition of Done verification for autonomous jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from packages.orchestrator.quality_gates import (
    artifact_path_exists,
    invalid_planning_artifact_paths,
)
from packages.schemas.jobs import JobRecord
from packages.schemas.models import ReviewDecision


@dataclass
class CompletionVerification:
    passed: bool
    missing_evidence: list[str] = field(default_factory=list)
    unresolved_findings: list[str] = field(default_factory=list)


class DefinitionOfDoneVerifier:
    """Check whether a job has enough evidence to enter DONE."""

    ALLOW_EMPTY_ARTIFACT_NAMES = frozenset({".gitkeep", ".keep", "__init__.py"})

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

        for artifact in self._required_artifacts(record):
            if invalid_planning_artifact_paths([artifact]):
                missing.append(f"required_artifact_invalid:{artifact}")
            elif self._artifact_is_non_file(record, artifact):
                missing.append(f"required_artifact_non_file:{artifact}")
            elif not self._artifact_exists(record, artifact):
                missing.append(f"required_artifact_missing:{artifact}")
            elif self._artifact_is_empty(record, artifact):
                missing.append(f"required_artifact_empty:{artifact}")
        for target in self._target_files(record):
            if invalid_planning_artifact_paths([target]):
                missing.append(f"target_file_invalid:{target}")
            elif self._artifact_is_non_file(record, target):
                missing.append(f"target_file_non_file:{target}")
            elif not self._artifact_exists(record, target):
                missing.append(f"target_file_missing:{target}")
            elif self._artifact_is_empty(record, target):
                missing.append(f"target_file_empty:{target}")

        test_run = outputs.get("test_run")
        if not isinstance(test_run, dict) or test_run.get("success") is not True:
            missing.append("unit_tests_success")
        elif not self._unit_tests_executed(record, test_run):
            missing.append("unit_tests_executed")
        if (
            self._runtime_required(record)
            or "runtime_smoke" in outputs
        ) and not self._success_value(outputs.get("runtime_smoke")):
            missing.append("runtime_smoke_success")
        if (
            self._acceptance_checks_required(record)
            or "acceptance_checks" in outputs
        ) and not self._success_value(outputs.get("acceptance_checks")):
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

    @classmethod
    def _runtime_required(cls, record: JobRecord) -> bool:
        return cls._metadata_has_non_empty(record, "runtime")

    @classmethod
    def _acceptance_checks_required(cls, record: JobRecord) -> bool:
        return cls._metadata_has_non_empty(record, "acceptance_checks")

    @classmethod
    def _test_evidence_required(cls, record: JobRecord) -> bool:
        return cls._metadata_has_non_empty(record, "require_test_evidence")

    @classmethod
    def _unit_tests_executed(cls, record: JobRecord, test_run: dict[str, Any]) -> bool:
        executed = test_run.get("executed_test_count")
        if isinstance(executed, int):
            return executed >= 1
        output = str(test_run.get("output_excerpt") or "").lower()
        if "no tests ran" in output:
            return False
        return not cls._test_evidence_required(record)

    @staticmethod
    def _metadata_has_non_empty(record: JobRecord, key: str) -> bool:
        metadata = record.spec.metadata if isinstance(record.spec.metadata, dict) else {}
        candidates: list[Any] = [metadata.get(key)]
        constraints = metadata.get("constraints")
        if isinstance(constraints, dict):
            candidates.append(constraints.get(key))
        for value in candidates:
            if isinstance(value, dict) and value:
                return True
            if isinstance(value, list) and value:
                return True
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, bool):
                return value
        return False

    @staticmethod
    def _required_artifacts(record: JobRecord) -> set[str]:
        paths = DefinitionOfDoneVerifier._collect_task_paths(
            record.outputs,
            "required_artifacts",
        )
        paths.update(
            DefinitionOfDoneVerifier._collect_metadata_paths(
                record,
                "required_artifacts",
                "source_required_artifacts",
                "implementation_required_artifacts",
                "test_required_artifacts",
            )
        )
        return paths

    @staticmethod
    def _target_files(record: JobRecord) -> set[str]:
        paths = DefinitionOfDoneVerifier._collect_task_paths(
            record.outputs,
            "target_files",
        )
        paths.update(
            DefinitionOfDoneVerifier._collect_metadata_paths(record, "target_files")
        )
        return paths

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
    def _collect_metadata_paths(record: JobRecord, *keys: str) -> set[str]:
        metadata = record.spec.metadata if isinstance(record.spec.metadata, dict) else {}
        candidates: list[Any] = []
        for key in keys:
            candidates.append(metadata.get(key))
        constraints = metadata.get("constraints")
        if isinstance(constraints, dict):
            for key in keys:
                candidates.append(constraints.get(key))

        paths: set[str] = set()
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                paths.add(candidate)
            elif isinstance(candidate, list):
                paths.update(
                    item for item in candidate if isinstance(item, str) and item.strip()
                )
        return paths

    @staticmethod
    def _artifact_exists(record: JobRecord, relative_path: str) -> bool:
        root = record.spec.workspace_root or record.spec.repo_path
        return artifact_path_exists(relative_path, workspace_root=root)

    @staticmethod
    def _artifact_is_non_file(record: JobRecord, relative_path: str) -> bool:
        if invalid_planning_artifact_paths([relative_path]):
            return False
        value = str(relative_path).replace("\\", "/").strip()
        normalized = PurePosixPath(value)
        workspace = Path(record.spec.workspace_root or record.spec.repo_path).resolve()
        target = (workspace / Path(*normalized.parts)).resolve()
        if workspace not in [target, *target.parents]:
            return False
        return target.exists() and not target.is_file()

    @classmethod
    def _artifact_is_empty(cls, record: JobRecord, relative_path: str) -> bool:
        if invalid_planning_artifact_paths([relative_path]):
            return False
        value = str(relative_path).replace("\\", "/").strip()
        normalized = PurePosixPath(value)
        if normalized.name in cls.ALLOW_EMPTY_ARTIFACT_NAMES:
            return False
        workspace = Path(record.spec.workspace_root or record.spec.repo_path).resolve()
        target = (workspace / Path(*normalized.parts)).resolve()
        if workspace not in [target, *target.parents] or not target.is_file():
            return False
        return target.stat().st_size == 0
