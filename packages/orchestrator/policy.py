"""Policy loading, risk classification, and tool enforcement."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from packages.orchestrator.workspace import WorkspacePolicy
from packages.schemas.approvals import PolicyAction, RiskDecision, RiskLevel


class ToolPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_policy: str = "deny"
    allow_by_role: dict[str, list[str]] = Field(default_factory=dict)


class GitPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch_prefix: str = "acos/"
    forbid_direct_main_write: bool = True
    forbid_force_push: bool = True


class SandboxPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    network_default: str = "disabled"
    timeout_seconds: int = 600
    max_memory: str = "8g"
    max_cpus: int = 4


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_from_job: bool = True
    allow_read: bool = True
    allow_write: bool = True
    allow_delete: bool = True
    max_delete_files_without_approval: int = 20
    max_patch_changed_files_without_approval: int = 50
    max_patch_deleted_lines_without_approval: int = 2000
    forbidden_path_patterns: list[str] = Field(default_factory=list)
    test_path_prefixes: list[str] = Field(default_factory=lambda: ["tests/"])
    dependency_manifest_files: list[str] = Field(default_factory=list)
    forbid_test_writes_by_role: list[str] = Field(default_factory=list)
    forbid_dependency_writes_by_role: list[str] = Field(default_factory=list)


class ApprovalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_channel: str = "console"
    request_ttl_minutes: int = 1440
    allow_cli_approval: bool = True
    allow_http_approval: bool = True
    allow_notification_links: bool = True
    require_signed_tokens: bool = True
    notify_on: list[str] = Field(default_factory=list)


class RiskRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_allow: list[str] = Field(default_factory=list)
    require_approval: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class AutonomyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "workspace_auto"
    level: int = 4
    default_workspace_action: str = "allow"
    require_approval_for_high_risk: bool = True
    deny_critical_operations: bool = True


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    autonomy: AutonomyPolicy = Field(default_factory=AutonomyPolicy)
    tools: ToolPolicy = Field(default_factory=ToolPolicy)
    git: GitPolicy = Field(default_factory=GitPolicy)
    sandbox: SandboxPolicy = Field(default_factory=SandboxPolicy)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    risk_rules: RiskRules = Field(default_factory=RiskRules)


class PolicyEngine:
    """Apply configured safety, workspace, and approval policies."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    @classmethod
    def from_path(cls, path: str | Path) -> "PolicyEngine":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(PolicyConfig(**raw))

    def build_workspace_policy(self, workspace_root: str | Path) -> WorkspacePolicy:
        return WorkspacePolicy(
            workspace_root=Path(workspace_root),
            forbidden_patterns=self.config.workspace.forbidden_path_patterns,
            max_delete_files_without_approval=self.config.workspace.max_delete_files_without_approval,
            max_patch_changed_files_without_approval=self.config.workspace.max_patch_changed_files_without_approval,
            max_patch_deleted_lines_without_approval=self.config.workspace.max_patch_deleted_lines_without_approval,
        )

    def is_tool_allowed(self, role: str, tool_name: str) -> bool:
        allowed = self.config.tools.allow_by_role.get(role, [])
        return tool_name in allowed

    def assert_tool_allowed(self, role: str, tool_name: str) -> None:
        if not self.is_tool_allowed(role, tool_name):
            raise PermissionError(f"Tool {tool_name} is not allowed for role {role}")

    def assert_release_commit_allowed(self, role: str) -> None:
        if role != "release_manager":
            raise PermissionError("Only release_manager may commit")

    def assert_branch_allowed(self, branch: str) -> None:
        if branch in {"main", "master", "develop"} and self.config.git.forbid_direct_main_write:
            raise PermissionError("Direct main/master/develop writes are forbidden")
        if not branch.startswith(self.config.git.branch_prefix):
            raise PermissionError(
                f"Branch {branch} must start with {self.config.git.branch_prefix}"
            )

    def normalize_path(self, path: str) -> str:
        normalized = PurePosixPath(path.replace("\\", "/")).as_posix()
        if normalized.startswith("./"):
            return normalized[2:]
        return normalized

    def is_forbidden_path(self, path: str) -> bool:
        normalized = self.normalize_path(path)
        return any(
            fnmatch(normalized, pattern)
            or fnmatch(PurePosixPath(normalized).name, pattern)
            for pattern in self.config.workspace.forbidden_path_patterns
        )

    def is_test_path(self, path: str) -> bool:
        normalized = self.normalize_path(path)
        basename = PurePosixPath(normalized).name
        return any(
            normalized.startswith(prefix) for prefix in self.config.workspace.test_path_prefixes
        ) or (basename.startswith("test_") and basename.endswith(".py"))

    def is_dependency_manifest(self, path: str) -> bool:
        normalized = self.normalize_path(path)
        basename = PurePosixPath(normalized).name
        return any(
            fnmatch(normalized, pattern) or fnmatch(basename, pattern)
            for pattern in self.config.workspace.dependency_manifest_files
        )

    def assert_patch_target_allowed(self, role: str, path: str) -> None:
        normalized = self.normalize_path(path)
        if self.is_forbidden_path(normalized):
            raise PermissionError(f"Path {normalized} is forbidden by policy")
        if (
            role in self.config.workspace.forbid_test_writes_by_role
            and self.is_test_path(normalized)
        ):
            raise PermissionError(f"Role {role} may not modify test files: {normalized}")
        if (
            role in self.config.workspace.forbid_dependency_writes_by_role
            and self.is_dependency_manifest(normalized)
        ):
            raise PermissionError(
                f"Role {role} may not modify dependency manifests: {normalized}"
            )

    def classify_named_operation(self, operation: str) -> RiskDecision:
        if operation in self.config.risk_rules.deny:
            return RiskDecision(
                operation=operation,
                policy_action=PolicyAction.DENY,
                risk_level=RiskLevel.CRITICAL,
                reason=f"{operation} is explicitly denied by policy",
            )
        if operation in self.config.risk_rules.require_approval:
            return RiskDecision(
                operation=operation,
                policy_action=PolicyAction.REQUIRE_APPROVAL,
                risk_level=RiskLevel.HIGH,
                reason=f"{operation} requires approval by policy",
            )
        if operation in {"notification_send_status", "git_commit_local"}:
            return RiskDecision(
                operation=operation,
                policy_action=PolicyAction.ALLOW_AND_AUDIT,
                risk_level=RiskLevel.MEDIUM,
                reason=f"{operation} is allowed with audit",
            )
        return RiskDecision(
            operation=operation,
            policy_action=PolicyAction.ALLOW,
            risk_level=RiskLevel.LOW,
            reason=f"{operation} is auto-allowed by policy",
        )

    def classify_tool_call(
        self,
        *,
        role: str,
        tool_name: str,
        arguments: dict[str, Any],
        workspace_root: str | Path,
        job_metadata: dict[str, Any] | None = None,
    ) -> RiskDecision:
        self.assert_tool_allowed(role, tool_name)
        workspace_policy = self.build_workspace_policy(workspace_root)
        if tool_name == "repo_server.read_file":
            return workspace_policy.classify_path_access(str(arguments.get("path", "")), "read")
        if tool_name == "repo_server.repo_tree":
            return self.classify_named_operation("workspace_file_read")
        if tool_name == "repo_server.search_text":
            return self.classify_named_operation("workspace_file_read")
        if tool_name == "repo_server.apply_patch":
            path = str(arguments.get("path", ""))
            self.assert_patch_target_allowed(role, path)
            return workspace_policy.classify_path_access(
                path,
                "patch",
                changed_files=int(arguments.get("changed_files", 1)),
                deleted_lines=arguments.get("deleted_lines"),
                new_content=str(arguments.get("content", "")),
            )
        if tool_name == "git_server.status":
            return self.classify_named_operation("git_status")
        if tool_name == "git_server.diff":
            return self.classify_named_operation("git_diff")
        if tool_name == "git_server.log_recent":
            return self.classify_named_operation("git_log")
        if tool_name == "git_server.current_branch":
            return self.classify_named_operation("git_log")
        if tool_name == "git_server.create_branch":
            branch = str(arguments.get("branch", ""))
            if branch in {"main", "master", "develop"}:
                return self.classify_named_operation("direct_main_write")
            if not branch.startswith(self.config.git.branch_prefix):
                return self.classify_named_operation("git_remote_write")
            return self.classify_named_operation("branch_create_acos_prefix")
        if tool_name == "git_server.commit":
            branch = str(arguments.get("branch", ""))
            if branch in {"main", "master", "develop"}:
                return self.classify_named_operation("direct_main_write")
            if not branch.startswith(self.config.git.branch_prefix):
                return self.classify_named_operation("git_remote_write")
            return self.classify_named_operation("git_commit_local")
        if tool_name == "test_server.run_test":
            command_name = str(arguments.get("command_name", ""))
            if command_name in {
                "",
                "auto",
                "django-test",
                "prepare-runtime-auto",
                "runtime-smoke-auto",
                "django-wsgi-check",
                "pytest",
            }:
                return self.classify_named_operation("test_run_allowlisted")
            if command_name == "npm-lint":
                return self.classify_named_operation("lint_run_allowlisted")
            if command_name == "npm-typecheck":
                return self.classify_named_operation("typecheck_run_allowlisted")
            if command_name in {"python-compile", "pytest-unit", "npm-test"}:
                return self.classify_named_operation("test_run_allowlisted")
            return self.classify_named_operation("arbitrary_shell")
        if tool_name == "test_server.install_package":
            constraints = (job_metadata or {}).get("constraints", {})
            if not isinstance(constraints, dict) or not constraints.get("allow_dependency_addition"):
                return self.classify_named_operation("package_install_non_allowlisted")
            return RiskDecision(
                operation="package_install_allowlisted_virtualenv",
                policy_action=PolicyAction.ALLOW_AND_AUDIT,
                risk_level=RiskLevel.MEDIUM,
                reason="job explicitly allows dependency installation inside the active virtualenv",
                details={"package": str(arguments.get("package", ""))},
            )
        if tool_name == "test_server.run_command":
            argv = arguments.get("argv", [])
            if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
                return self.classify_named_operation("arbitrary_shell")
            lowered = [item.lower() for item in argv]
            joined = " ".join(lowered)
            if lowered[0] == "sudo":
                return self.classify_named_operation("sudo")
            if lowered[0] in {"curl", "wget", "ssh", "scp", "rsync"}:
                return self.classify_named_operation("external_network_non_allowlisted")
            destructive_tokens = (
                " drop ",
                " dropdb",
                " flush",
                " sqlflush",
                " db:drop",
                " db:reset",
                " db:rollback",
                " migrate:rollback",
                " downgrade",
            )
            if any(token in f" {joined} " for token in destructive_tokens):
                return self.classify_named_operation("destructive_db_migration")
            return RiskDecision(
                operation="workspace_runtime_exec",
                policy_action=PolicyAction.ALLOW_AND_AUDIT,
                risk_level=RiskLevel.MEDIUM,
                reason="workspace-local runtime command execution is allowed with audit",
                details={"argv": argv, "mode": str(arguments.get("mode", "oneshot"))},
            )
        if tool_name == "memory_server.read_memory" or tool_name == "memory_server.search_memory":
            return self.classify_named_operation("memory_read_project")
        if tool_name == "memory_server.write_memory" or tool_name == "memory_server.update_task_summary":
            return self.classify_named_operation("memory_write_project")
        if tool_name == "notify_server.send_notification":
            return self.classify_named_operation("notification_send_status")
        if tool_name in {
            "notify_server.send_approval_request",
            "notify_server.send_runtime_wait",
            "notify_server.send_provider_recovered",
            "notify_server.send_job_completed",
            "notify_server.send_job_failed",
        }:
            return RiskDecision(
                operation="notification_send_status",
                policy_action=PolicyAction.ALLOW_AND_AUDIT,
                risk_level=RiskLevel.MEDIUM,
                reason="notification is allowed with audit",
            )
        return RiskDecision(
            operation=tool_name,
            policy_action=PolicyAction.ALLOW_AND_AUDIT,
            risk_level=RiskLevel.MEDIUM,
            reason=f"{tool_name} is not explicitly classified; allowing with audit",
        )

    def list_allowed_tools(self, role: str | None = None) -> dict[str, list[str]] | list[str]:
        if role is None:
            return {
                role_name: sorted(tool_names)
                for role_name, tool_names in self.config.tools.allow_by_role.items()
            }
        return sorted(self.config.tools.allow_by_role.get(role, []))
