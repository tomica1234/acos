"""Fake MCP servers used for tests and local demos."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from packages.memory.redaction import redact_text
from packages.memory.store import SQLiteMemoryStore
from packages.mcp_client.router import MCPRouter
from packages.schemas.agent_outputs import TestRunResult

FORBIDDEN_PATH_PARTS = {
    ".env",
    ".ssh",
    ".aws",
    "id_rsa",
    "id_ed25519",
    "token",
    "tokens",
}

TEST_COMMAND_ALLOWLIST: dict[str, list[str]] = {
    "django-test": [sys.executable, "manage.py", "test"],
    "python-compile": [sys.executable, "-m", "compileall", "."],
    "pytest": [sys.executable, "-m", "pytest", "-q"],
    "pytest-unit": [sys.executable, "-m", "pytest", "tests", "-q"],
    "npm-test": ["npm", "test"],
    "npm-lint": ["npm", "run", "lint"],
    "npm-typecheck": ["npm", "run", "typecheck"],
}


class RepoServer:
    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.modified_files: set[str] = set()

    def _resolve(self, relative_path: str) -> Path:
        if any(part in FORBIDDEN_PATH_PARTS for part in Path(relative_path).parts):
            raise ValueError("forbidden path access")
        target = (self.workspace_root / relative_path).resolve()
        if self.workspace_root not in [target, *target.parents]:
            raise ValueError("workspace escape detected")
        if target.is_symlink():
            resolved_target = target.resolve()
            if self.workspace_root not in [resolved_target, *resolved_target.parents]:
                raise ValueError("symlink escape detected")
        return target

    def repo_tree(self) -> dict[str, object]:
        files = []
        for path in sorted(self.workspace_root.rglob("*")):
            if not path.is_file():
                continue
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if any(part.startswith(".") for part in path.relative_to(self.workspace_root).parts):
                continue
            if path.suffix in {".sqlite3", ".db", ".pyc"}:
                continue
            files.append(str(path.relative_to(self.workspace_root)))
        return {"files": files}

    def read_file(self, path: str, max_chars: int = 20000) -> dict[str, object]:
        file_path = self._resolve(path)
        if file_path.suffix in {".sqlite3", ".db", ".pyc"}:
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
            executed_test_count=_parse_executed_test_count(output),
        )
        return payload.model_dump()


def _parse_executed_test_count(output: str) -> int | None:
    lowered = output.lower()
    if "no tests ran" in lowered or "collected 0 items" in lowered:
        return 0
    ran_match = re.search(r"\bran\s+(\d+)\s+tests?\b", lowered)
    if ran_match:
        return int(ran_match.group(1))
    collected_match = re.search(r"\bcollected\s+(\d+)\s+items?\b", lowered)
    if collected_match and "passed" not in lowered and "failed" not in lowered:
        return int(collected_match.group(1))
    counts = [
        int(match.group(1))
        for match in re.finditer(
            r"\b(\d+)\s+(?:passed|failed|errors?|skipped|xfailed|xpassed)\b",
            lowered,
        )
    ]
    return sum(counts) if counts else None


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
        resolved_scope = self._resolve_read_scope(uri=uri, scope=scope)
        return {"entries": self.memory_store.read(scope=resolved_scope, limit=limit)}

    def search_memory(self, query: str, uri: str | None = None, scope: str | None = None, limit: int = 20) -> dict[str, object]:
        resolved_scope = self._resolve_read_scope(uri=uri, scope=scope)
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
            return scope_part or "global", key_part or "default"
        return scope or "global", item_key or "default"

    @staticmethod
    def _resolve_read_scope(
        *,
        uri: str | None = None,
        scope: str | None = None,
    ) -> str | None:
        if uri is None:
            return scope
        if not uri.startswith("memory://"):
            raise ValueError("memory URI must start with memory://")
        remainder = uri[len("memory://") :]
        scope_part, _, _key_part = remainder.partition("/")
        return scope_part or None


class NotifyServer:
    def __init__(self) -> None:
        self.notifications: list[str] = []

    def send_notification(self, body: str | None = None, message: str | None = None) -> dict[str, object]:
        text = redact_text((body if body is not None else message or "")[:2000])
        self.notifications.append(text)
        return {"message": text, "channel": "console"}


class FakeMCPEnvironment:
    """Convenience wrapper bundling fake servers and a router."""

    def __init__(
        self,
        workspace_root: str | Path,
        memory_db_path: str | Path,
        scripted_test_results: Iterable[TestRunResult] | None = None,
    ) -> None:
        self.repo_server = RepoServer(workspace_root)
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
        return router
