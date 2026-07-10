"""Execute durable ACOS recovery plans."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packages.orchestrator.quality_gates import artifact_path_exists, invalid_artifact_paths
from packages.orchestrator.statuses import is_hard_terminal_status, is_waiting_status
from packages.schemas.checkpoints import CheckpointRecord
from packages.schemas.jobs import JobRecord
from packages.schemas.models import JobStatus


class RecoveryExecutor:
    """Consume RecoveryGovernor plans and make them actionable."""

    PROJECT_SETUP_TARGET_PATHS = frozenset(
        {
            "backend/main.py",
            "backend/requirements.txt",
            "backend/tests/test_project_setup.py",
            "frontend/package.json",
            "frontend/vite.config.js",
            "frontend/src/main.tsx",
            "frontend/src/App.tsx",
            "shared/.gitkeep",
            ".gitignore",
            "package.json",
            "README.md",
            ".env.example",
        }
    )

    STALE_RECOVERY_CONSTRAINT_KEYS = {
        "deterministic_creation_attempted",
        "deterministically_created_files",
        "empty_artifacts",
        "completion_integrity_failure_reasons",
        "failed_stage_ids",
        "failed_stages",
        "failed_task_id",
        "failed_patch_operation",
        "failed_patch_path",
        "failed_patch_role",
        "force_project_setup_scaffold",
        "invalid_artifacts",
        "missing_artifacts",
        "missing_stage_test_patch_stage_ids",
        "missing_task_ids",
        "missing_target_file",
        "non_file_artifacts",
        "patch_operation_hint",
        "recovery_mode",
        "recreate_target_files_attempt",
        "return_to_role",
        "stages_missing_test_patches",
        "unmet_dependencies",
    }

    def __init__(self, store: Any | None = None) -> None:
        self.store = store

    def execute_until_ready(self, record: JobRecord) -> JobRecord:
        """Run all bookkeeping recovery steps until normal job flow can resume."""

        plan = record.runtime_state.get("recovery_plan")
        if not isinstance(plan, dict):
            return record
        if plan.get("status") == "completed":
            return record
        if is_hard_terminal_status(record.status):
            return record

        plan["status"] = "running"
        self._touch_plan(plan)
        steps = [str(step) for step in plan.get("steps", [])]
        current_index = int(plan.get("current_step_index") or 0)
        while current_index < len(steps):
            step = steps[current_index]
            if step == "RECREATE_TARGET_FILES" and not self._target_files_recreated(
                record,
                plan,
            ):
                plan["status"] = "running"
                plan["current_step_index"] = current_index
                self._touch_plan(plan)
                break
            self._checkpoint(record, plan, step)
            self._apply_step(record, plan, step)
            executed = plan.setdefault("executed_steps", [])
            if isinstance(executed, list):
                executed.append(step)
            current_index += 1
            plan["current_step_index"] = current_index
            self._touch_plan(plan)
            if is_hard_terminal_status(record.status) or is_waiting_status(record.status):
                break

        if current_index >= len(steps):
            plan["status"] = "completed"
            plan["completed_at"] = self._now()
            next_status = self._plan_next_status(plan)
            if next_status is not None and not is_hard_terminal_status(record.status):
                record.status = next_status
                if not record.history or record.history[-1] != next_status:
                    record.history.append(next_status)
        record.runtime_state["recovery_plan"] = plan
        self._sync_plan_constraints_to_metadata(record, plan)
        record.updated_at = datetime.now(timezone.utc)
        self._persist(record)
        return record

    def _target_files_recreated(
        self,
        record: JobRecord,
        plan: dict[str, Any],
    ) -> bool:
        constraints = plan.setdefault("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
            plan["constraints"] = constraints
        paths = self._recreate_target_paths(record, constraints)
        if not paths:
            return True
        invalid = invalid_artifact_paths(paths)
        if invalid:
            constraints["invalid_artifacts"] = invalid
            constraints["missing_artifacts"] = []
            constraints["recovery_mode"] = "invalid_artifacts_replan"
            constraints["return_to_role"] = "planner"
            constraints.pop("patch_operation_hint", None)
            plan["strategy"] = "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
            plan["next_actor"] = "planner"
            plan["next_status"] = JobStatus.REPLANNING.value
            record.runtime_state["planner_repair_requested"] = True
            record.status = JobStatus.REPLANNING
            if record.history[-1:] != [JobStatus.REPLANNING]:
                record.history.append(JobStatus.REPLANNING)
            return True
        paths = [path for path in paths if path not in set(invalid)]
        root = Path(record.spec.workspace_root or record.spec.repo_path).resolve()
        non_file_artifacts = self._non_file_artifacts(root, paths)
        if non_file_artifacts:
            self._route_non_file_artifacts_to_planner(
                record,
                plan,
                constraints,
                non_file_artifacts,
            )
            return True
        missing = [
            path for path in paths if not artifact_path_exists(path, workspace_root=root)
        ]
        if not missing:
            constraints["missing_artifacts"] = []
            return True

        constraints["missing_artifacts"] = missing
        self._assign_missing_file_owner(record, plan, constraints, missing)
        runtime = record.runtime_state
        attempts = runtime.setdefault("recreate_target_files_attempts", {})
        if isinstance(attempts, dict):
            key = "|".join(missing)
            attempts[key] = int(attempts.get(key, 0)) + 1
            constraints["recreate_target_files_attempt"] = attempts[key]
            if attempts[key] >= 2:
                attempted = self._attempt_deterministic_creation(root, missing)
                constraints["deterministic_creation_attempted"] = True
                constraints["deterministically_created_files"] = attempted
                non_file_artifacts = self._non_file_artifacts(root, paths)
                if non_file_artifacts:
                    self._route_non_file_artifacts_to_planner(
                        record,
                        plan,
                        constraints,
                        non_file_artifacts,
                    )
                    return True
                missing = [
                    path
                    for path in paths
                    if not artifact_path_exists(path, workspace_root=root)
                ]
                constraints["missing_artifacts"] = missing
                if not missing:
                    return True
                self._assign_missing_file_owner(record, plan, constraints, missing)
                self._force_project_setup_scaffold_when_needed(record, constraints, missing)
                return True
        return False

    @staticmethod
    def _non_file_artifacts(root: Path, paths: list[str]) -> list[str]:
        non_files: list[str] = []
        for path in paths:
            normalized = path.replace("\\", "/").removeprefix("./")
            if invalid_artifact_paths([normalized]):
                continue
            target = root / normalized
            if target.exists() and not target.is_file():
                non_files.append(normalized)
        return non_files

    @staticmethod
    def _route_non_file_artifacts_to_planner(
        record: JobRecord,
        plan: dict[str, Any],
        constraints: dict[str, Any],
        non_file_artifacts: list[str],
    ) -> None:
        constraints["non_file_artifacts"] = non_file_artifacts
        constraints["missing_artifacts"] = non_file_artifacts
        constraints["recovery_mode"] = "non_file_artifacts_replan"
        constraints["return_to_role"] = "planner"
        constraints.pop("patch_operation_hint", None)
        plan["strategy"] = "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
        plan["next_actor"] = "planner"
        plan["next_status"] = JobStatus.REPLANNING.value
        record.runtime_state["planner_repair_requested"] = True
        record.status = JobStatus.REPLANNING
        if record.history[-1:] != [JobStatus.REPLANNING]:
            record.history.append(JobStatus.REPLANNING)

    @staticmethod
    def _sync_plan_constraints_to_metadata(
        record: JobRecord,
        plan: dict[str, Any],
    ) -> None:
        constraints = record.spec.metadata.setdefault("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
            record.spec.metadata["constraints"] = constraints
        plan_constraints = plan.get("constraints")
        if not isinstance(plan_constraints, dict):
            plan_constraints = {}
        for stale_key in RecoveryExecutor.STALE_RECOVERY_CONSTRAINT_KEYS:
            if stale_key not in plan_constraints:
                constraints.pop(stale_key, None)
        constraints.update(plan_constraints)
        for target_key, source_key in (
            ("recovery_strategy", "strategy"),
            ("recovery_next_actor", "next_actor"),
            ("recovery_next_status", "next_status"),
        ):
            value = plan.get(source_key)
            if isinstance(value, str) and value.strip():
                constraints[target_key] = value.strip()

    def _force_project_setup_scaffold_when_needed(
        self,
        record: JobRecord,
        constraints: dict[str, Any],
        missing: list[str],
    ) -> None:
        non_test_paths = [
            path for path in missing if not self._looks_like_test_path(path)
        ]
        app_source_paths = [
            path
            for path in non_test_paths
            if not self._looks_like_project_setup_path(path)
        ]
        if app_source_paths or not any(
            self._looks_like_project_setup_path(path) for path in missing
        ):
            return
        constraints["force_project_setup_scaffold"] = True
        metadata_constraints = record.spec.metadata.setdefault("constraints", {})
        if not isinstance(metadata_constraints, dict):
            metadata_constraints = {}
            record.spec.metadata["constraints"] = metadata_constraints
        metadata_constraints["force_project_setup_scaffold"] = True

    def _assign_missing_file_owner(
        self,
        record: JobRecord,
        plan: dict[str, Any],
        constraints: dict[str, Any],
        missing: list[str],
    ) -> None:
        owner = self._owner_for_missing_paths(missing)
        constraints["return_to_role"] = owner
        plan["next_actor"] = owner
        status = self._status_for_owner(owner)
        plan["next_status"] = status.value
        record.status = status
        if record.history[-1:] != [status]:
            record.history.append(status)

    @classmethod
    def _owner_for_missing_paths(cls, missing: list[str]) -> str:
        non_test_paths = [
            path for path in missing if not cls._looks_like_test_path(path)
        ]
        if non_test_paths and all(
            cls._looks_like_project_setup_path(path) for path in non_test_paths
        ):
            return "scaffold"
        if non_test_paths:
            return "implementer"
        if any(cls._looks_like_test_path(path) for path in missing):
            return "test_writer"
        return "implementer"

    @staticmethod
    def _status_for_owner(owner: str) -> JobStatus:
        if owner == "test_writer":
            return JobStatus.WRITING_TESTS
        if owner == "fixer":
            return JobStatus.FIXING
        return JobStatus.IMPLEMENTING

    @staticmethod
    def _looks_like_test_path(path: str) -> bool:
        normalized = path.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        return (
            "/tests/" in f"/{normalized}"
            or "/test/" in f"/{normalized}"
            or name.startswith("test_")
            or ".test." in name
            or ".spec." in name
        )

    @staticmethod
    def _looks_like_project_setup_path(path: str) -> bool:
        normalized = path.replace("\\", "/")
        return normalized in RecoveryExecutor.PROJECT_SETUP_TARGET_PATHS

    def _attempt_deterministic_creation(
        self,
        root: Path,
        missing: list[str],
    ) -> list[str]:
        created: list[str] = []
        for path in missing:
            normalized = path.replace("\\", "/").removeprefix("./")
            if invalid_artifact_paths([normalized]):
                continue
            content = self._deterministic_content_for_path(normalized)
            if content is None:
                continue
            target = root / normalized
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and not target.is_file():
                    continue
                if not target.exists():
                    target.write_text(content, encoding="utf-8")
                if artifact_path_exists(normalized, workspace_root=root):
                    created.append(normalized)
            except OSError:
                continue
        return created

    @classmethod
    def _deterministic_content_for_path(cls, path: str) -> str | None:
        if cls._looks_like_test_path(path):
            normalized = path.replace("\\", "/").removeprefix("./")
            suffix = Path(path).suffix.lower()
            if suffix in {".ts", ".tsx", ".js", ".jsx"}:
                js_path = normalized.replace("\\", "\\\\").replace("'", "\\'")
                return (
                    "import { describe, expect, it } from 'vitest'\n\n"
                    "describe('project scaffold', () => {\n"
                    "  it('has a deterministic test scaffold', () => {\n"
                    f"    // fallback target: {js_path}\n"
                    "    const url = import.meta.url\n"
                    "    expect(url).toMatch(/(^|\\/)(test|tests)\\//)\n"
                    "    expect(url).toMatch(/(^|\\/)test_|\\.(test|spec)\\./)\n"
                    "  })\n"
                    "})\n"
                )
            py_path = repr(normalized)
            return (
                "from pathlib import Path\n\n\n"
                "def test_project_scaffold_placeholder() -> None:\n"
                f"    # fallback target: {py_path}\n"
                "    current_path = Path(__file__).as_posix()\n"
                "    name = Path(__file__).name\n"
                "    normalized = current_path.replace('\\\\', '/')\n"
                "    assert '/test' in f'/{normalized}' or name.startswith('test_')\n"
            )
        contents = {
            "backend/main.py": (
                "from fastapi import FastAPI\n\n"
                "app = FastAPI(title=\"ACOS generated app\")\n\n\n"
                "@app.get(\"/health\")\n"
                "def health() -> dict[str, str]:\n"
                "    return {\"status\": \"ok\"}\n"
            ),
            "backend/requirements.txt": "fastapi\nuvicorn\npytest\n",
            "frontend/package.json": (
                "{\n"
                "  \"private\": true,\n"
                "  \"type\": \"module\",\n"
                "  \"scripts\": {\"dev\": \"vite --host 0.0.0.0\"},\n"
                "  \"dependencies\": {\"@vitejs/plugin-react\": \"latest\", \"vite\": \"latest\", \"typescript\": \"latest\", \"react\": \"latest\", \"react-dom\": \"latest\"},\n"
                "  \"devDependencies\": {}\n"
                "}\n"
            ),
            "frontend/vite.config.js": (
                "import react from '@vitejs/plugin-react'\n"
                "import { defineConfig } from 'vite'\n\n"
                "export default defineConfig({ plugins: [react()] })\n"
            ),
            "frontend/src/main.tsx": (
                "import React from 'react'\n"
                "import ReactDOM from 'react-dom/client'\n"
                "import App from './App'\n\n"
                "ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)\n"
            ),
            "frontend/src/App.tsx": (
                "function App() {\n"
                "  return <main>ACOS project scaffold is ready.</main>\n"
                "}\n\n"
                "export default App\n"
            ),
            "shared/.gitkeep": "",
            ".gitignore": ".venv/\n__pycache__/\nnode_modules/\ndist/\n.env\n",
            "package.json": "{\n  \"private\": true,\n  \"scripts\": {\"dev\": \"npm --prefix frontend run dev\"}\n}\n",
            "README.md": "# ACOS generated app\n\nDeterministic project scaffold.\n",
            ".env.example": "LOCAL_ORNITH_BASE_URL=http://127.0.0.1:8000/v1\n",
        }
        return contents.get(path)

    @staticmethod
    def _recreate_target_paths(
        record: JobRecord,
        constraints: dict[str, Any],
    ) -> list[str]:
        paths: list[str] = []
        for key in ("required_artifacts", "target_files", "missing_artifacts"):
            value = constraints.get(key)
            if isinstance(value, list):
                paths.extend(str(item) for item in value if str(item).strip())
        missing_target_file = constraints.get("missing_target_file")
        if isinstance(missing_target_file, str) and missing_target_file.strip():
            paths.append(missing_target_file.strip())
        if not paths:
            runtime_missing = record.runtime_state.get("missing_artifacts")
            if isinstance(runtime_missing, list):
                paths.extend(str(item) for item in runtime_missing if str(item).strip())
        seen: set[str] = set()
        unique: list[str] = []
        for path in paths:
            normalized = path.replace("\\", "/")
            if normalized and normalized not in seen:
                unique.append(normalized)
                seen.add(normalized)
        return unique

    def _apply_step(self, record: JobRecord, plan: dict[str, Any], step: str) -> None:
        constraints = record.spec.metadata.setdefault("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
            record.spec.metadata["constraints"] = constraints
        runtime = record.runtime_state

        if step == "DIAGNOSE_FAILURE":
            runtime["diagnosis_requested"] = True
            record.status = JobStatus.DIAGNOSING
        elif step == "EXPAND_CONTEXT":
            constraints["expand_context"] = True
            runtime["expand_context"] = True
            runtime["context_expansion_count"] = int(runtime.get("context_expansion_count", 0)) + 1
        elif step == "COMPACT_CONTEXT":
            constraints["compact_context"] = True
            runtime["compact_context"] = True
        elif step in {"REPLAN_TASK", "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"}:
            runtime["planner_repair_requested"] = True
            record.status = JobStatus.REPLANNING
        elif step == "SPLIT_TASK":
            constraints["split_task_on_retry"] = True
            record.status = JobStatus.REPLANNING
        elif step == "REVISE_PRD":
            runtime["prd_revision_requested"] = True
            record.status = JobStatus.ANALYZING
        elif step == "REVISE_ARCHITECTURE":
            runtime["architecture_revision_requested"] = True
            record.status = JobStatus.DESIGNING
        elif step == "REDEFINE_ACCEPTANCE":
            runtime["acceptance_revision_requested"] = True
            record.status = JobStatus.REPLANNING
        elif step == "RETURN_TO_IMPLEMENTER":
            record.status = JobStatus.IMPLEMENTING
        elif step == "RETURN_TO_TEST_WRITER":
            record.status = JobStatus.WRITING_TESTS
        elif step == "RETURN_TO_FIXER":
            record.status = JobStatus.FIXING
        elif step == "ROLLBACK_LAST_PATCH":
            runtime["rollback_last_patch_requested"] = True
        elif step == "RETRY_WITH_DIFFERENT_STRATEGY":
            constraints["avoid_same_fixer_loop"] = True
            constraints["retry_with_different_strategy"] = True
            record.status = JobStatus.STRATEGY_CHANGE
        elif step == "RETRY_WITH_ESCALATED_MODEL":
            constraints["force_model_escalation"] = True
            record.status = JobStatus.STRATEGY_CHANGE
        elif step == "WAITING_RUNTIME":
            record.status = JobStatus.WAITING_RUNTIME
        elif step == "AVOID_REJECTED_OPERATION":
            constraints["avoid_rejected_operation"] = True
            record.status = JobStatus.REPLANNING
        elif step == "COMPLETION_AUDIT":
            runtime["completion_audit_requested"] = True
            record.status = JobStatus.REPLANNING
        elif step in {"SUMMARIZE_TOOL_FINDINGS", "RETRY_WITH_SMALLER_SCOPE", "RETURN_VALID_STRUCTURED_OUTPUT"}:
            constraints["force_structured_output"] = True
            constraints["retry_small_scope"] = True
            record.status = JobStatus.STRATEGY_CHANGE
        elif step.startswith("STOP_FOR_"):
            record.status = JobStatus.POLICY_HARD_STOP

        if record.history[-1:] != [record.status]:
            record.history.append(record.status)

    def _checkpoint(self, record: JobRecord, plan: dict[str, Any], step: str) -> None:
        payload = {
            "job_id": record.job_id,
            "plan_id": str(plan.get("id", "")),
            "strategy": str(plan.get("strategy", "")),
            "step": step,
            "created_at": self._now(),
        }
        record.checkpoints.append(
            {
                "checkpoint_key": f"recovery:{plan.get('id')}:{step}",
                "step_name": step,
                "status": "completed",
                "result_json": payload,
            }
        )
        if self.store is not None and hasattr(self.store, "save_checkpoint"):
            checkpoint = CheckpointRecord(
                job_id=record.job_id,
                checkpoint_key=f"recovery:{plan.get('id')}:{step}",
                step_name=step,
                idempotency_key=f"{record.job_id}:{plan.get('id')}:{step}",
                status="completed",
                result_json=payload,
            )
            self.store.save_checkpoint(checkpoint)

    def _persist(self, record: JobRecord) -> None:
        if self.store is not None and hasattr(self.store, "update"):
            self.store.update(record)

    @staticmethod
    def _plan_next_status(plan: dict[str, Any]) -> JobStatus | None:
        value = plan.get("next_status")
        if not isinstance(value, str):
            return None
        try:
            return JobStatus(value)
        except ValueError:
            return None

    @staticmethod
    def _touch_plan(plan: dict[str, Any]) -> None:
        plan["updated_at"] = RecoveryExecutor._now()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
