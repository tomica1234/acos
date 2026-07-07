"""Fake MCP servers used for tests and local demos."""

from __future__ import annotations

import importlib.metadata
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tomllib
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
            files.append(path.relative_to(self.workspace_root).as_posix())
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

    def search_text(
        self,
        query: str,
        glob: str | None = None,
        max_results: int = 50,
        context_lines: int = 2,
        case_sensitive: bool = False,
        regex: bool = False,
    ) -> dict[str, object]:
        matches: list[dict[str, object]] = []
        needle = query if case_sensitive else query.lower()
        pattern = re.compile(query, 0 if case_sensitive else re.IGNORECASE) if regex else None
        for relative_path in self.repo_tree()["files"]:
            if glob and not Path(str(relative_path)).match(glob):
                continue
            file_path = self._resolve(str(relative_path))
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            haystack = content if case_sensitive else content.lower()
            if pattern is None and needle not in haystack:
                continue
            if pattern is not None and pattern.search(content) is None:
                continue
            lines = content.splitlines()
            for index, line in enumerate(lines, start=1):
                search_line = line if case_sensitive else line.lower()
                if pattern is None and needle not in search_line:
                    continue
                if pattern is not None and pattern.search(line) is None:
                    continue
                start = max(1, index - context_lines)
                end = min(len(lines), index + context_lines)
                snippet = "\n".join(lines[start - 1 : end])
                matches.append(
                    {
                        "path": str(relative_path),
                        "line": index,
                        "line_number": index,
                        "before": lines[start - 1 : index - 1],
                        "match": line,
                        "after": lines[index:end],
                        "snippet": snippet,
                    }
                )
                if len(matches) >= max_results:
                    return {"matches": matches}
        return {"matches": matches}

    def apply_patch(
        self,
        path: str,
        content: str | None = None,
        operation: str = "update",
        new_path: str | None = None,
        unified_diff: str | None = None,
        base_sha256: str | None = None,
        expected_old_content: str | None = None,
        executable: bool | None = None,
    ) -> dict[str, object]:
        file_path = self._resolve(path)
        old_content: str | None = None
        old_sha256: str | None = None
        if file_path.exists() and file_path.is_file():
            old_content = file_path.read_text(encoding="utf-8")
            old_sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if base_sha256 is not None and file_path.exists():
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if digest != base_sha256:
                raise ValueError("base_sha256 mismatch")
        if expected_old_content is not None:
            if not file_path.exists():
                raise ValueError("expected_old_content provided but file does not exist")
            if file_path.read_text(encoding="utf-8") != expected_old_content:
                raise ValueError("expected_old_content mismatch")
        if operation == "create":
            if content is None:
                raise ValueError("create operation requires content")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists():
                raise ValueError(f"create operation would overwrite existing file: {path}")
            file_path.write_text(content, encoding="utf-8")
        elif operation == "update":
            if not file_path.exists():
                raise ValueError(f"target_files_missing:update target does not exist: {path}")
            if unified_diff is not None:
                patched = self._apply_unified_diff(
                    old_content if old_content is not None else "",
                    unified_diff,
                )
                file_path.write_text(patched, encoding="utf-8")
            elif content is not None:
                file_path.write_text(content, encoding="utf-8")
            else:
                raise ValueError("update operation requires content or unified_diff")
        elif operation == "delete":
            if not file_path.exists():
                raise ValueError(f"delete operation target does not exist: {path}")
            file_path.unlink()
        elif operation == "rename":
            if not new_path:
                raise ValueError("rename operation requires new_path")
            if not file_path.exists():
                raise ValueError(f"rename operation source does not exist: {path}")
            new_file_path = self._resolve(new_path)
            if new_file_path.exists():
                raise ValueError(f"rename operation target already exists: {new_path}")
            new_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.rename(new_file_path)
            self.modified_files.add(new_path)
        else:
            raise ValueError(f"unsupported patch operation: {operation}")
        if executable is not None and operation != "delete":
            target_path = self._resolve(new_path if operation == "rename" and new_path else path)
            mode = target_path.stat().st_mode
            if executable:
                target_path.chmod(mode | 0o111)
            else:
                target_path.chmod(mode & ~0o111)
        self.modified_files.add(path)
        return {
            "path": path,
            "operation": operation,
            "new_path": new_path,
            "rollback": {
                "old_sha256": old_sha256,
                "old_content": old_content,
            },
        }

    @staticmethod
    def _apply_unified_diff(original: str, unified_diff: str) -> str:
        original_lines = original.splitlines(keepends=True)
        result: list[str] = []
        source_index = 0
        lines = unified_diff.splitlines(keepends=True)
        index = 0
        while index < len(lines):
            line = lines[index]
            if line.startswith(("--- ", "+++ ")):
                index += 1
                continue
            if not line.startswith("@@"):
                index += 1
                continue
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if match is None:
                raise ValueError("patch_conflict: invalid unified diff hunk")
            old_start = int(match.group(1)) - 1
            if old_start < source_index:
                raise ValueError("patch_conflict: overlapping unified diff hunk")
            result.extend(original_lines[source_index:old_start])
            source_index = old_start
            index += 1
            while index < len(lines) and not lines[index].startswith("@@"):
                hunk_line = lines[index]
                marker = hunk_line[:1]
                text = hunk_line[1:]
                if marker == " ":
                    if source_index >= len(original_lines) or original_lines[source_index] != text:
                        raise ValueError("patch_conflict: context mismatch")
                    result.append(original_lines[source_index])
                    source_index += 1
                elif marker == "-":
                    if source_index >= len(original_lines) or original_lines[source_index] != text:
                        raise ValueError("patch_conflict: removal mismatch")
                    source_index += 1
                elif marker == "+":
                    result.append(text)
                elif hunk_line.startswith("\\ No newline"):
                    pass
                else:
                    raise ValueError("patch_conflict: unsupported unified diff line")
                index += 1
        result.extend(original_lines[source_index:])
        return "".join(result)


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


class RealGitServer:
    """Real git backend with protected branch and rollback safeguards."""

    def __init__(self, workspace_root: str | Path, *, branch_prefix: str = "acos/") -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.branch_prefix = branch_prefix

    def status(self) -> dict[str, object]:
        output = self._git("status", "--porcelain=v1")
        modified = [
            line[3:]
            for line in output.splitlines()
            if len(line) >= 4 and line[:2].strip()
        ]
        return {"modified_files": modified, "branch": self.current_branch()["branch"]}

    def diff(self) -> dict[str, object]:
        return {"diff": self._git("diff")}

    def current_branch(self) -> dict[str, object]:
        return {"branch": self._git("rev-parse", "--abbrev-ref", "HEAD").strip()}

    def create_branch(self, branch: str) -> dict[str, object]:
        self._assert_branch_allowed(branch)
        self._git("checkout", "-B", branch)
        return {"branch": branch}

    def commit(self, message: str, branch: str) -> dict[str, object]:
        self._assert_branch_allowed(branch)
        if not message.startswith("acos:"):
            raise ValueError("commit message must start with 'acos:'")
        current = self.current_branch()["branch"]
        if current != branch:
            raise ValueError(f"current branch {current} does not match requested branch {branch}")
        self._git("add", "--all")
        self._git("commit", "-m", message)
        sha = self._git("rev-parse", "HEAD").strip()
        return {"sha": sha, "message": message, "branch": branch}

    def rollback_last_acos_commit(self) -> dict[str, object]:
        message = self._git("log", "-1", "--pretty=%s").strip()
        if not message.startswith("acos:"):
            raise ValueError("last commit is not an ACOS commit")
        self._git("revert", "--no-edit", "HEAD")
        return {"reverted": True, "message": message}

    def restore_file(self, path: str) -> dict[str, object]:
        resolved = (self.workspace_root / path).resolve()
        if self.workspace_root not in [resolved, *resolved.parents]:
            raise ValueError("workspace escape detected")
        self._git("restore", "--", path)
        return {"path": path, "restored": True}

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.workspace_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git failed")
        return result.stdout

    def _assert_branch_allowed(self, branch: str) -> None:
        if branch in {"main", "master", "develop"}:
            raise ValueError("direct protected branch operation is forbidden")
        if not branch.startswith(self.branch_prefix):
            raise ValueError(f"branch must start with {self.branch_prefix!r}")


class TestServer:
    def __init__(
        self,
        workspace_root: str | Path,
        scripted_results: Iterable[TestRunResult] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.scripted_results = list(scripted_results or [])
        self._installed_dependency_roots: set[Path] = set()

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
        dependency_result = self._ensure_project_test_dependencies(timeout_seconds)
        if dependency_result is not None:
            return dependency_result.model_dump()
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

    def _ensure_project_test_dependencies(
        self,
        timeout_seconds: int,
    ) -> TestRunResult | None:
        inferred_result = self._install_missing_requirements(
            self.workspace_root,
            self._inferred_test_requirements(),
            timeout_seconds,
        )
        if inferred_result is not None:
            return inferred_result
        dependency_roots = self._dependency_roots()
        if not dependency_roots:
            return None
        for root in dependency_roots:
            if root in self._installed_dependency_roots:
                continue
            install_command = self._dependency_install_command(root)
            if install_command is None:
                self._installed_dependency_roots.add(root)
                continue
            install_result = self._run_dependency_install(
                root,
                install_command,
                timeout_seconds,
            )
            if install_result is not None:
                return install_result
            self._installed_dependency_roots.add(root)
        return None

    def _install_missing_requirements(
        self,
        root: Path,
        requirements: list[str],
        timeout_seconds: int,
    ) -> TestRunResult | None:
        missing = [
            requirement
            for requirement in requirements
            if not self._requirement_installed(requirement)
        ]
        if not missing:
            return None
        return self._run_dependency_install(
            root,
            [sys.executable, "-m", "pip", "install", *missing],
            timeout_seconds,
        )

    @staticmethod
    def _run_dependency_install(
        root: Path,
        install_command: list[str],
        timeout_seconds: int,
    ) -> TestRunResult | None:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            result = subprocess.run(
                install_command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = "\n".join(
                item
                for item in [
                    "Dependency installation timed out before tests could run.",
                    str(exc.stdout or ""),
                    str(exc.stderr or ""),
                ]
                if item
            )
            return TestRunResult(
                success=False,
                command=install_command,
                failed_tests=[],
                output_excerpt=output[-20000:],
                exit_code=124,
                executed_test_count=0,
            )
        output = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode != 0:
            return TestRunResult(
                success=False,
                command=install_command,
                failed_tests=[],
                output_excerpt=(
                    "Dependency installation failed before tests could run.\n"
                    + output
                )[-20000:],
                exit_code=result.returncode,
                executed_test_count=0,
            )
        return None

    def _dependency_roots(self) -> list[Path]:
        roots: set[Path] = set()
        for path in self.workspace_root.rglob("requirements.txt"):
            if self._is_visible_workspace_file(path):
                roots.add(path.parent)
        for path in self.workspace_root.rglob("pyproject.toml"):
            if self._is_visible_workspace_file(path) and self._pyproject_has_dependencies(path):
                roots.add(path.parent)
        return sorted(roots, key=lambda item: len(item.parts))

    def _is_visible_workspace_file(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.workspace_root)
        except ValueError:
            return False
        return not any(part.startswith(".") for part in relative.parts)

    def _dependency_install_command(self, root: Path) -> list[str] | None:
        requirements = root / "requirements.txt"
        if requirements.exists():
            missing = [
                requirement
                for requirement in self._read_requirements(requirements)
                if not self._requirement_installed(requirement)
            ]
            if missing:
                return [sys.executable, "-m", "pip", "install", *missing]
            return None
        pyproject = root / "pyproject.toml"
        if pyproject.exists() and self._pyproject_has_dependencies(pyproject):
            return [sys.executable, "-m", "pip", "install", "-e", str(root)]
        return None

    def _inferred_test_requirements(self) -> list[str]:
        requirements: list[str] = []
        for path in self.workspace_root.rglob("*.py"):
            if not self._is_visible_workspace_file(path):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if (
                "python -m playwright" in content
                or '"-m", "playwright"' in content
                or "'-m', 'playwright'" in content
                or "import playwright" in content
            ):
                requirements.append("playwright")
            if "import pytest_asyncio" in content or "pytest.mark.asyncio" in content:
                requirements.append("pytest-asyncio")
            if "aiosqlite" in content:
                requirements.append("aiosqlite")
        return sorted(set(requirements))

    @staticmethod
    def _read_requirements(path: Path) -> list[str]:
        requirements: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#") or item.startswith("-"):
                continue
            requirements.append(item)
        return requirements

    @staticmethod
    def _requirement_installed(requirement: str) -> bool:
        package_name = re.split(r"[<>=!~;\\[]", requirement, maxsplit=1)[0].strip()
        if not package_name:
            return True
        try:
            importlib.metadata.version(package_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    @staticmethod
    def _pyproject_has_dependencies(path: Path) -> bool:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return False
        project = data.get("project")
        if not isinstance(project, dict):
            return False
        dependencies = project.get("dependencies")
        optional = project.get("optional-dependencies")
        return bool(dependencies) or bool(optional)


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
