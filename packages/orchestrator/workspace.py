"""Workspace sandbox policy helpers."""

from __future__ import annotations

import difflib
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

from packages.schemas.approvals import PolicyAction, RiskDecision, RiskLevel


class WorkspacePolicy:
    """Classify workspace path access against sandbox and approval rules."""

    def __init__(
        self,
        workspace_root: Path,
        forbidden_patterns: list[str],
        *,
        max_delete_files_without_approval: int = 20,
        max_patch_changed_files_without_approval: int = 50,
        max_patch_deleted_lines_without_approval: int = 2000,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.forbidden_patterns = list(forbidden_patterns)
        self.max_delete_files_without_approval = max_delete_files_without_approval
        self.max_patch_changed_files_without_approval = max_patch_changed_files_without_approval
        self.max_patch_deleted_lines_without_approval = max_patch_deleted_lines_without_approval

    def normalize_path(self, path: str) -> str:
        normalized = PurePosixPath(path.replace("\\", "/")).as_posix()
        if normalized.startswith("./"):
            return normalized[2:]
        return normalized

    def classify_path_access(
        self,
        path: str,
        operation: str,
        *,
        delete_count: int = 1,
        changed_files: int = 1,
        deleted_lines: int | None = None,
        new_content: str | None = None,
    ) -> RiskDecision:
        normalized = self.normalize_path(path)
        try:
            target = self._resolve_path(normalized, operation=operation)
        except ValueError as exc:
            return RiskDecision(
                operation="workspace_escape",
                policy_action=PolicyAction.DENY,
                risk_level=RiskLevel.CRITICAL,
                reason=str(exc),
                details={"path": normalized, "requested_operation": operation},
            )
        if self._matches_forbidden_pattern(normalized):
            return RiskDecision(
                operation="secret_file_read" if operation == "read" else "credential_store_access",
                policy_action=PolicyAction.DENY,
                risk_level=RiskLevel.CRITICAL,
                reason=f"path {normalized} matches a forbidden pattern",
                details={"path": normalized, "requested_operation": operation},
            )
        if operation == "delete":
            if delete_count > self.max_delete_files_without_approval or normalized in {"", ".", "/"}:
                return RiskDecision(
                    operation="mass_delete",
                    policy_action=PolicyAction.REQUIRE_APPROVAL,
                    risk_level=RiskLevel.HIGH,
                    reason="delete count exceeds auto-allow threshold",
                    details={"path": normalized, "delete_count": delete_count},
                )
            return RiskDecision(
                operation="workspace_file_delete",
                policy_action=PolicyAction.ALLOW,
                risk_level=RiskLevel.LOW,
                reason="workspace delete is within the auto-allow threshold",
                details={"path": normalized},
            )
        if operation == "patch":
            effective_deleted_lines = (
                deleted_lines
                if deleted_lines is not None
                else self._estimate_deleted_lines(target, new_content or "")
            )
            if (
                changed_files > self.max_patch_changed_files_without_approval
                or effective_deleted_lines > self.max_patch_deleted_lines_without_approval
            ):
                return RiskDecision(
                    operation="large_patch",
                    policy_action=PolicyAction.REQUIRE_APPROVAL,
                    risk_level=RiskLevel.HIGH,
                    reason="patch size exceeds auto-allow threshold",
                    details={
                        "path": normalized,
                        "changed_files": changed_files,
                        "deleted_lines": effective_deleted_lines,
                    },
                )
            return RiskDecision(
                operation="workspace_patch_apply",
                policy_action=PolicyAction.ALLOW,
                risk_level=RiskLevel.LOW,
                reason="workspace patch is within auto-allow threshold",
                details={"path": normalized, "deleted_lines": effective_deleted_lines},
            )
        if operation == "read":
            return RiskDecision(
                operation="workspace_file_read",
                policy_action=PolicyAction.ALLOW,
                risk_level=RiskLevel.LOW,
                reason="workspace file read is auto-allowed",
                details={"path": normalized},
            )
        if operation in {"write", "create"}:
            return RiskDecision(
                operation="workspace_file_write",
                policy_action=PolicyAction.ALLOW,
                risk_level=RiskLevel.LOW,
                reason="workspace file write is auto-allowed",
                details={"path": normalized},
            )
        return RiskDecision(
            operation=f"workspace_{operation}",
            policy_action=PolicyAction.ALLOW_AND_AUDIT,
            risk_level=RiskLevel.MEDIUM,
            reason="unknown workspace operation allowed with audit",
            details={"path": normalized},
        )

    def _resolve_path(self, path: str, *, operation: str) -> Path:
        candidate = PurePosixPath(path)
        if candidate.is_absolute():
            target = Path(path).resolve()
        else:
            if any(part in {"..", "."} for part in candidate.parts):
                raise ValueError("relative traversal outside workspace is forbidden")
            target = (self.workspace_root / candidate).resolve()
        if self.workspace_root not in [target, *target.parents]:
            raise ValueError("workspace escape detected")
        if target.exists() and target.is_symlink():
            resolved_target = target.resolve()
            if self.workspace_root not in [resolved_target, *resolved_target.parents]:
                raise ValueError("symlink escape detected")
        if operation == "delete" and target == self.workspace_root:
            raise ValueError("deleting the workspace root is forbidden")
        return target

    def _matches_forbidden_pattern(self, path: str) -> bool:
        basename = PurePosixPath(path).name
        return any(fnmatch(path, pattern) or fnmatch(basename, pattern) for pattern in self.forbidden_patterns)

    @staticmethod
    def _estimate_deleted_lines(target: Path, new_content: str) -> int:
        if not target.exists():
            return 0
        try:
            original = target.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return 0
        proposed = new_content.splitlines()
        diff = difflib.unified_diff(original, proposed, lineterm="")
        deleted = 0
        for line in diff:
            if line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("-"):
                deleted += 1
        return deleted
