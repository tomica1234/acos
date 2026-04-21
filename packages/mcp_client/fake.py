"""Fake MCP servers used for tests and local demos."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable

from packages.memory.redaction import redact_text
from packages.memory.store import SQLiteMemoryStore
from packages.mcp_client.router import MCPRouter
from packages.orchestrator.workspace import WorkspacePolicy
from packages.schemas.agent_outputs import TestRunResult

SAFE_HIDDEN_FILE_NAMES = {
    ".env.example",
    ".env.sample",
    ".env.template",
    ".gitignore",
    ".python-version",
}

FORBIDDEN_EXACT_NAMES = {
    ".git",
    ".ssh",
    ".aws",
    "authorized_keys",
    "credentials",
    "id_rsa",
    "id_ed25519",
    "known_hosts",
}

FORBIDDEN_NAME_PATTERN = re.compile(
    r"(^|[._-])(token|secret|credential|password|passwd)([._-]|$)",
    re.IGNORECASE,
)
FORBIDDEN_SUFFIXES = {".pem", ".key", ".p12", ".crt", ".cer"}
BINARY_SUFFIXES = {".sqlite3", ".db", ".pyc"}
MAX_READ_CHARS = 50000
MAX_PATCH_CHARS = 200000
MAX_TEST_TIMEOUT_SECONDS = 600

TEST_COMMAND_ALLOWLIST: dict[str, list[str]] = {
    "python-compile": [sys.executable, "-m", "compileall", "."],
    "pytest": [sys.executable, "-m", "pytest", "-q"],
    "pytest-unit": [sys.executable, "-m", "pytest", "tests", "-q"],
    "npm-test": ["npm", "test"],
    "npm-lint": ["npm", "run", "lint"],
    "npm-typecheck": ["npm", "run", "typecheck"],
}


class RepoServer:
    def __init__(
        self,
        workspace_root: str | Path,
        workspace_policy: WorkspacePolicy | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.modified_files: set[str] = set()
        self.workspace_policy = workspace_policy

    @staticmethod
    def _normalize_relative_path(relative_path: str) -> PurePosixPath:
        normalized = PurePosixPath(relative_path.replace("\\", "/"))
        if normalized.is_absolute():
            raise ValueError("absolute paths are forbidden")
        if not normalized.parts:
            raise ValueError("empty path is forbidden")
        if any(part in {"..", "."} for part in normalized.parts):
            raise ValueError("relative traversal is forbidden")
        return normalized

    @classmethod
    def _assert_component_allowed(cls, part: str, *, is_leaf: bool) -> None:
        lowered = part.lower()
        if part in FORBIDDEN_EXACT_NAMES:
            raise ValueError("forbidden path access")
        if lowered.startswith(".env") and part not in SAFE_HIDDEN_FILE_NAMES:
            raise ValueError("forbidden path access")
        if part.startswith(".") and not (is_leaf and part in SAFE_HIDDEN_FILE_NAMES):
            raise ValueError("hidden path access is forbidden")
        if FORBIDDEN_NAME_PATTERN.search(part):
            raise ValueError("forbidden path access")
        if Path(part).suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError("forbidden path access")

    def _resolve(self, relative_path: str) -> Path:
        normalized = self._normalize_relative_path(relative_path)
        if self.workspace_policy is not None:
            decision = self.workspace_policy.classify_path_access(
                normalized.as_posix(),
                "read",
            )
            if decision.policy_action.value == "deny":
                raise ValueError(decision.reason)
        for index, part in enumerate(normalized.parts):
            self._assert_component_allowed(part, is_leaf=index == len(normalized.parts) - 1)
        target = (self.workspace_root / normalized).resolve()
        if self.workspace_root not in [target, *target.parents]:
            raise ValueError("workspace escape detected")
        return target

    def repo_tree(self) -> dict[str, object]:
        files = []
        for path in sorted(self.workspace_root.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            relative_path = path.relative_to(self.workspace_root).as_posix()
            try:
                self._resolve(relative_path)
            except ValueError:
                continue
            if path.suffix in BINARY_SUFFIXES:
                continue
            files.append(relative_path)
        return {"files": files}

    def read_file(self, path: str, max_chars: int = 20000) -> dict[str, object]:
        if max_chars <= 0 or max_chars > MAX_READ_CHARS:
            raise ValueError(f"max_chars must be between 1 and {MAX_READ_CHARS}")
        file_path = self._resolve(path)
        if file_path.suffix in BINARY_SUFFIXES:
            raise ValueError("binary file access is forbidden")
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError("binary file access is forbidden")
        content = content[:max_chars]
        return {"path": path, "content": content}

    def search_text(self, query: str) -> dict[str, object]:
        matches: list[dict[str, str]] = []
        for relative_path in self.repo_tree()["files"]:
            file_path = self._resolve(str(relative_path))
            content = file_path.read_text(encoding="utf-8")
            if query in content:
                matches.append({"path": str(relative_path), "snippet": query})
        return {"matches": matches}

    def apply_patch(self, path: str, content: str, operation: str = "update") -> dict[str, object]:
        if operation not in {"create", "update"}:
            raise ValueError("unsupported patch operation")
        if len(content) > MAX_PATCH_CHARS:
            raise ValueError(f"patch content exceeds {MAX_PATCH_CHARS} chars")
        file_path = self._resolve(path)
        if operation == "create":
            file_path.parent.mkdir(parents=True, exist_ok=True)
        elif not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        self.modified_files.add(path)
        return {"path": path, "operation": operation}


class GitServer:
    def __init__(self, repo_server: RepoServer) -> None:
        self.repo_server = repo_server
        self.commits: list[dict[str, object]] = []
        self._current_branch = "acos/default"

    def diff(self) -> dict[str, object]:
        fragments: list[str] = []
        for path in sorted(self.repo_server.modified_files):
            content = self.repo_server.read_file(path)["content"]
            fragments.append(f"--- {path}\n{content}")
        return {"diff": "\n".join(fragments)}

    def status(self) -> dict[str, object]:
        return {
            "modified_files": sorted(self.repo_server.modified_files),
            "commit_count": len(self.commits),
            "branch": self._current_branch,
        }

    def current_branch(self) -> dict[str, object]:
        return {"branch": self._current_branch}

    def create_branch(self, branch: str) -> dict[str, object]:
        self._assert_branch_allowed(branch)
        self._current_branch = branch
        return {"branch": branch}

    def commit(self, message: str, branch: str) -> dict[str, object]:
        self._assert_branch_allowed(branch)
        if not message.startswith("acos:"):
            raise ValueError("commit message must start with 'acos:'")
        snapshot = self.status()
        self.commits.append({"message": message, "branch": branch, "snapshot": snapshot})
        self.repo_server.modified_files.clear()
        self._current_branch = branch
        return {"commit_index": len(self.commits), "message": message, "branch": branch}

    def log_recent(self, limit: int = 10) -> dict[str, object]:
        return {"commits": self.commits[-limit:]}

    @staticmethod
    def _assert_branch_allowed(branch: str) -> None:
        if branch in {"main", "master", "develop"}:
            raise ValueError("direct protected branch operation is forbidden")
        if not branch.startswith("acos/"):
            raise ValueError("branch must start with 'acos/'")
        if any(char in branch for char in {" ", "\t", "\n", "\r", ":", "~", "^", "?"}):
            raise ValueError("invalid branch name")
        if ".." in branch:
            raise ValueError("invalid branch name")


class TestServer:
    def __init__(
        self,
        workspace_root: str | Path,
        scripted_results: Iterable[TestRunResult] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.scripted_results = list(scripted_results or [])

    def run_test(
        self,
        command_name: str = "pytest",
        timeout_seconds: int = 120,
    ) -> dict[str, object]:
        if self.scripted_results:
            return self.scripted_results.pop(0).model_dump()
        if command_name not in TEST_COMMAND_ALLOWLIST:
            raise ValueError(f"command_name {command_name} is not allowlisted")
        if timeout_seconds <= 0 or timeout_seconds > MAX_TEST_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be between 1 and {MAX_TEST_TIMEOUT_SECONDS}"
            )
        command = list(TEST_COMMAND_ALLOWLIST[command_name])
        if len(command) >= 3 and command[0] == sys.executable and command[1] != "-B":
            command = [command[0], "-B", *command[1:]]
        for cache_dir in self.workspace_root.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(self.workspace_root) if not existing else f"{self.workspace_root}{os.pathsep}{existing}"
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            command,
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        failed_tests = [
            line.strip()
            for line in output.splitlines()
            if "::" in line and ("FAILED" in line or "ERROR" in line)
        ]
        payload = TestRunResult(
            success=result.returncode == 0,
            command=command,
            failed_tests=failed_tests,
            output_excerpt=output[-20000:],
            exit_code=result.returncode,
        )
        return payload.model_dump()


TestServer.__test__ = False


class MemoryServer:
    MAX_CONTENT_CHARS = 8000

    def __init__(self, memory_store: SQLiteMemoryStore) -> None:
        self.memory_store = memory_store

    def write_memory(
        self,
        uri: str | None = None,
        content: str | None = None,
        *,
        scope: str | None = None,
        item_key: str | None = None,
        value: str | None = None,
    ) -> dict[str, object]:
        memory_scope, memory_key = self._resolve_scope(uri=uri, scope=scope, item_key=item_key)
        body = redact_text((content if content is not None else value or "")[: self.MAX_CONTENT_CHARS])
        self.memory_store.write(memory_scope, memory_key, body)
        return {"scope": memory_scope, "key": memory_key}

    def read_memory(self, uri: str | None = None, scope: str | None = None, limit: int = 20) -> dict[str, object]:
        resolved_scope = self._resolve_scope(uri=uri, scope=scope)[0]
        return {"entries": self.memory_store.read(scope=resolved_scope, limit=limit)}

    def search_memory(self, query: str, uri: str | None = None, scope: str | None = None, limit: int = 20) -> dict[str, object]:
        resolved_scope = self._resolve_scope(uri=uri, scope=scope)[0]
        return {"entries": self.memory_store.search(query=query, scope=resolved_scope, limit=limit)}

    def update_task_summary(self, uri: str, summary: str) -> dict[str, object]:
        scope, item_key = self._resolve_scope(uri=uri)
        self.memory_store.update_task_summary(scope=scope, task_key=item_key, summary=summary)
        return {"scope": scope, "key": item_key}

    @staticmethod
    def _resolve_scope(
        *,
        uri: str | None = None,
        scope: str | None = None,
        item_key: str | None = None,
    ) -> tuple[str | None, str]:
        if uri is not None:
            if not uri.startswith("memory://"):
                raise ValueError("memory URI must start with memory://")
            remainder = uri[len("memory://") :]
            scope_part, _, key_part = remainder.partition("/")
            return scope_part or None, key_part or "default"
        return scope, item_key or "default"


class NotifyServer:
    def __init__(self) -> None:
        self.notifications: list[str] = []
        self.approval_notifications: list[dict[str, str]] = []
        self.runtime_notifications: list[dict[str, str]] = []
        self.job_notifications: list[dict[str, str]] = []

    def send_notification(self, body: str | None = None, message: str | None = None) -> dict[str, object]:
        text = redact_text((body if body is not None else message or "")[:2000])
        self.notifications.append(text)
        return {"message": text, "channel": "console"}

    def send_approval_request(
        self,
        approval_id: str,
        job_id: str,
        risk_level: str,
        operation: str,
        reason: str,
        approve_url: str | None = None,
        reject_url: str | None = None,
        cli_command: str | None = None,
    ) -> dict[str, object]:
        stored_message = redact_text(
            "\n".join(
                [
                    "ACOS approval required",
                    f"Job: {job_id}",
                    f"Operation: {operation}",
                    f"Risk: {risk_level}",
                    f"Reason: {reason}",
                    f"Approve: {approve_url or 'use CLI'}",
                    f"Reject: {reject_url or 'use CLI'}",
                    f"CLI: {cli_command or f'acos approvals approve {approval_id}'}",
                ]
            )
        )
        self.notifications.append(stored_message)
        payload = {
            "approval_id": approval_id,
            "job_id": job_id,
            "risk_level": risk_level,
            "operation": operation,
            "reason": redact_text(reason),
            "approve_url": approve_url,
            "reject_url": reject_url,
            "cli_command": cli_command or f"acos approvals approve {approval_id}",
            "channel": "console",
        }
        self.approval_notifications.append(
            {
                "approval_id": approval_id,
                "job_id": job_id,
                "operation": operation,
                "risk_level": risk_level,
                "approve_url": approve_url or "",
                "reject_url": reject_url or "",
                "cli_command": cli_command or f"acos approvals approve {approval_id}",
            }
        )
        return payload

    def send_runtime_wait(
        self,
        job_id: str,
        provider_key: str,
        model_key: str | None = None,
        reason: str | None = None,
        kind: str = "runtime_wait",
        channel: str = "console",
        cli_command: str | None = None,
    ) -> dict[str, object]:
        payload = {
            "job_id": job_id,
            "provider_key": provider_key,
            "model_key": model_key,
            "reason": redact_text(reason or "provider unavailable"),
            "kind": kind,
            "channel": channel,
            "cli_command": cli_command or f"acos jobs resume {job_id}",
        }
        self.runtime_notifications.append({key: str(value or "") for key, value in payload.items()})
        self.notifications.append(
            redact_text(
                "\n".join(
                    [
                        "ACOS runtime is waiting for model provider",
                        f"Job: {job_id}",
                        f"Provider: {provider_key}",
                        f"Model: {model_key or '-'}",
                        f"Reason: {reason or 'provider unavailable'}",
                        f"CLI: {payload['cli_command']}",
                    ]
                )
            )
        )
        return payload

    def send_provider_recovered(
        self,
        job_id: str,
        provider_key: str,
        model_key: str | None = None,
        cli_command: str | None = None,
    ) -> dict[str, object]:
        payload = {
            "job_id": job_id,
            "provider_key": provider_key,
            "model_key": model_key,
            "kind": "provider_recovered",
            "channel": "console",
            "cli_command": cli_command or f"acos jobs resume {job_id}",
        }
        self.runtime_notifications.append({key: str(value or "") for key, value in payload.items()})
        self.notifications.append(
            redact_text(
                f"ACOS provider recovered\nJob: {job_id}\nProvider: {provider_key}\nCLI: {payload['cli_command']}"
            )
        )
        return payload

    def send_job_completed(
        self,
        job_id: str,
        message: str | None = None,
    ) -> dict[str, object]:
        payload = {
            "job_id": job_id,
            "kind": "job_completed",
            "channel": "console",
            "message": redact_text(message or "job completed"),
        }
        self.job_notifications.append({key: str(value or "") for key, value in payload.items()})
        self.notifications.append(payload["message"])
        return payload

    def send_job_failed(
        self,
        job_id: str,
        message: str | None = None,
    ) -> dict[str, object]:
        payload = {
            "job_id": job_id,
            "kind": "job_failed",
            "channel": "console",
            "message": redact_text(message or "job failed"),
        }
        self.job_notifications.append({key: str(value or "") for key, value in payload.items()})
        self.notifications.append(payload["message"])
        return payload


class FakeMCPEnvironment:
    """Convenience wrapper bundling fake servers and a router."""

    def __init__(
        self,
        workspace_root: str | Path,
        memory_db_path: str | Path,
        scripted_test_results: Iterable[TestRunResult] | None = None,
        workspace_policy: WorkspacePolicy | None = None,
    ) -> None:
        self.repo_server = RepoServer(workspace_root, workspace_policy=workspace_policy)
        self.git_server = GitServer(self.repo_server)
        self.test_server = TestServer(workspace_root, scripted_results=scripted_test_results)
        self.memory_server = MemoryServer(SQLiteMemoryStore(memory_db_path))
        self.notify_server = NotifyServer()

    def build_router(self) -> MCPRouter:
        router = MCPRouter()
        router.register("repo_server.repo_tree", self.repo_server.repo_tree)
        router.register("repo_server.read_file", self.repo_server.read_file)
        router.register("repo_server.search_text", self.repo_server.search_text)
        router.register("repo_server.apply_patch", self.repo_server.apply_patch)
        router.register("git_server.diff", self.git_server.diff)
        router.register("git_server.status", self.git_server.status)
        router.register("git_server.current_branch", self.git_server.current_branch)
        router.register("git_server.create_branch", self.git_server.create_branch)
        router.register("git_server.commit", self.git_server.commit)
        router.register("git_server.log_recent", self.git_server.log_recent)
        router.register("test_server.run_test", self.test_server.run_test)
        router.register("memory_server.write_memory", self.memory_server.write_memory)
        router.register("memory_server.read_memory", self.memory_server.read_memory)
        router.register("memory_server.search_memory", self.memory_server.search_memory)
        router.register("memory_server.update_task_summary", self.memory_server.update_task_summary)
        router.register("notify_server.send_notification", self.notify_server.send_notification)
        router.register("notify_server.send_approval_request", self.notify_server.send_approval_request)
        router.register("notify_server.send_runtime_wait", self.notify_server.send_runtime_wait)
        router.register("notify_server.send_provider_recovered", self.notify_server.send_provider_recovered)
        router.register("notify_server.send_job_completed", self.notify_server.send_job_completed)
        router.register("notify_server.send_job_failed", self.notify_server.send_job_failed)
        return router
