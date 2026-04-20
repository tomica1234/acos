"""Policy loading and enforcement."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


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


class AutonomyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: int = 4


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    autonomy: AutonomyPolicy = Field(default_factory=AutonomyPolicy)
    tools: ToolPolicy = Field(default_factory=ToolPolicy)
    git: GitPolicy = Field(default_factory=GitPolicy)
    sandbox: SandboxPolicy = Field(default_factory=SandboxPolicy)
    blocked_operations: list[str] = Field(default_factory=list)


class PolicyEngine:
    """Apply configured safety and tool policies."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    @classmethod
    def from_path(cls, path: str | Path) -> "PolicyEngine":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(PolicyConfig(**raw))

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
        if branch in {"main", "master"} and self.config.git.forbid_direct_main_write:
            raise PermissionError("Direct main/master writes are forbidden")
        if not branch.startswith(self.config.git.branch_prefix):
            raise PermissionError(
                f"Branch {branch} must start with {self.config.git.branch_prefix}"
            )

    def assert_operation_allowed(self, operation: str) -> None:
        if operation in self.config.blocked_operations:
            raise PermissionError(f"Blocked operation: {operation}")

    def list_allowed_tools(self, role: str | None = None) -> dict[str, list[str]] | list[str]:
        if role is None:
            return {
                role_name: sorted(tool_names)
                for role_name, tool_names in self.config.tools.allow_by_role.items()
            }
        return sorted(self.config.tools.allow_by_role.get(role, []))
