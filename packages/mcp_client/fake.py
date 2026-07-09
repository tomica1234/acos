"""Fake MCP servers used for tests and local demos."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from fnmatch import fnmatch
from http.cookiejar import CookieJar
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterable

from packages.memory.redaction import redact_text
from packages.memory.store import SQLiteMemoryStore
from packages.mcp_client.router import MCPRouter
from packages.schemas.agent_outputs import TestRunResult
from packages.schemas.runtime import RuntimeHttpCheck

if TYPE_CHECKING:
    from packages.orchestrator.workspace import WorkspacePolicy

FORBIDDEN_PATH_PARTS = {
    ".env",
    ".ssh",
    ".aws",
    "id_rsa",
    "id_ed25519",
    "token",
    "tokens",
}
FORBIDDEN_PATH_PATTERNS = {
    ".env",
    ".env.local",
    ".env.*.local",
    ".env.development",
    ".env.*.development",
    ".env.production",
    ".env.*.production",
    ".env.test",
    ".env.*.test",
    ".git",
    ".git/**",
    "**/.git/**",
    "**/.ssh/**",
    "**/.aws/**",
    "**/id_rsa",
    "**/id_ed25519",
    "**/*credential*",
    "**/*secret*",
    "**/*token*",
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

MAX_TEST_TIMEOUT_SECONDS = 1200
MAX_INSTALL_TIMEOUT_SECONDS = 900
MAX_RUNTIME_STARTUP_WAIT_SECONDS = 10
PACKAGE_SPEC_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9_,.-]+\])?([<>=!~]{1,2}[A-Za-z0-9.*+!-]+(,[<>=!~]{1,2}[A-Za-z0-9.*+!-]+)*)?$"
)
DJANGO_SETTINGS_PATTERN = re.compile(
    r"setdefault\(\s*['\"]DJANGO_SETTINGS_MODULE['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)"
)
CSRF_INPUT_PATTERN = re.compile(
    r"<input[^>]*(?:name=['\"]csrfmiddlewaretoken['\"][^>]*value=['\"]([^'\"]+)['\"]|value=['\"]([^'\"]+)['\"][^>]*name=['\"]csrfmiddlewaretoken['\"])",
    re.IGNORECASE,
)
FASTAPI_APP_PATTERN = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*FastAPI\(")
FLASK_APP_PATTERN = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*Flask\(")
PYTHON_WEB_ENTRYPOINT_CANDIDATES = (
    "main.py",
    "app.py",
    "api.py",
    "app/main.py",
    "src/main.py",
    "src/app.py",
    "src/api.py",
)


class _PythonExecutable(str):
    """String path that compares equal to native and POSIX spellings on Windows."""

    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, str):
            return False
        raw = str(self)
        return other in {raw, Path(raw).as_posix()}


class RepoServer:
    def __init__(
        self,
        workspace_root: str | Path,
        workspace_policy: WorkspacePolicy | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.modified_files: set[str] = set()
        self.workspace_policy = workspace_policy

    def _resolve(self, relative_path: str) -> Path:
        normalized = PurePosixPath(str(relative_path).replace("\\", "/")).as_posix()
        if self.workspace_policy is not None:
            decision = self.workspace_policy.classify_path_access(normalized, "read")
            if decision.policy_action.value == "deny":
                raise ValueError(decision.reason)
        if self._is_forbidden_path(normalized):
            raise ValueError("forbidden path access")
        target = (self.workspace_root / normalized).resolve()
        if self.workspace_root not in [target, *target.parents]:
            raise ValueError("workspace escape detected")
        if target.is_symlink():
            resolved_target = target.resolve()
            if self.workspace_root not in [resolved_target, *resolved_target.parents]:
                raise ValueError("symlink escape detected")
        return target

    @staticmethod
    def _is_forbidden_path(path: str) -> bool:
        normalized = PurePosixPath(path).as_posix()
        parts = PurePosixPath(normalized).parts
        basename = PurePosixPath(normalized).name
        return (
            any(part in FORBIDDEN_PATH_PARTS for part in parts)
            or any(
                fnmatch(normalized, pattern) or fnmatch(basename, pattern)
                for pattern in FORBIDDEN_PATH_PATTERNS
            )
        )

    def repo_tree(self) -> dict[str, object]:
        files = []
        for path in sorted(self.workspace_root.rglob("*")):
            if path.is_symlink():
                continue
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
        command_name: str = "auto",
        timeout_seconds: int = 120,
        http_checks: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        if self.scripted_results:
            return self.scripted_results.pop(0).model_dump()
        if command_name == "auto":
            command_name = self._detect_test_command_name()
        if command_name == "prepare-runtime-auto":
            return self._run_runtime_prepare(timeout_seconds=timeout_seconds)
        if command_name == "runtime-smoke-auto":
            return self._run_runtime_smoke(
                timeout_seconds=timeout_seconds,
                http_checks=http_checks,
            )
        if command_name == "django-wsgi-check":
            return self._run_django_wsgi_check(timeout_seconds=timeout_seconds)
        if command_name not in TEST_COMMAND_ALLOWLIST:
            raise ValueError(f"command_name {command_name} is not allowlisted")
        if timeout_seconds <= 0 or timeout_seconds > MAX_TEST_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be between 1 and {MAX_TEST_TIMEOUT_SECONDS}"
            )
        command = list(TEST_COMMAND_ALLOWLIST[command_name])
        command = self._normalize_python_command(command)
        dependency_result = self._ensure_project_test_dependencies(timeout_seconds)
        if dependency_result is not None:
            return dependency_result.model_dump()
        return self._execute_test_command(command, timeout_seconds=timeout_seconds)

    def install_package(
        self,
        package: str,
        timeout_seconds: int = 600,
    ) -> dict[str, object]:
        package_spec = package.strip()
        if not PACKAGE_SPEC_PATTERN.fullmatch(package_spec):
            raise ValueError("package spec is not allowlisted")
        if timeout_seconds <= 0 or timeout_seconds > MAX_INSTALL_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be between 1 and {MAX_INSTALL_TIMEOUT_SECONDS}"
            )
        if not self._is_virtualenv_python():
            raise ValueError("package installation is only allowed inside an active virtualenv")
        command = self._normalize_python_command(
            [sys.executable, "-m", "pip", "install", package_spec]
        )
        completed = self._run_subprocess(command, timeout_seconds=timeout_seconds)
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return {
            "package": package_spec,
            "command": command,
            "success": completed.returncode == 0,
            "output_excerpt": output[-20000:],
            "exit_code": completed.returncode,
        }

    def run_command(
        self,
        argv: list[str],
        timeout_seconds: int = 120,
        mode: str = "oneshot",
        port: int | None = None,
        http_path: str = "/",
        http_checks: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        if timeout_seconds <= 0 or timeout_seconds > MAX_TEST_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be between 1 and {MAX_TEST_TIMEOUT_SECONDS}"
            )
        if mode not in {"oneshot", "server"}:
            raise ValueError("mode must be oneshot or server")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item.strip() for item in argv):
            raise ValueError("argv must be a non-empty list of strings")
        if not http_path.startswith("/"):
            raise ValueError("http_path must start with '/'")
        normalized_http_checks = (
            [RuntimeHttpCheck.model_validate(item).model_dump(exclude_none=True) for item in http_checks]
            if http_checks is not None
            else None
        )
        if normalized_http_checks is not None and mode != "server":
            raise ValueError("http_checks are only supported in server mode")
        resolved_port = port
        if mode == "server":
            if resolved_port is None:
                if not any("{port}" in item for item in argv):
                    raise ValueError("server mode argv must include a {port} placeholder or an explicit port")
                resolved_port = self._reserve_tcp_port()
            command = self._normalize_runtime_command(argv, port=resolved_port)
            return self._run_listening_process_check(
                command,
                port=resolved_port,
                timeout_seconds=timeout_seconds,
                http_path=http_path,
                http_checks=normalized_http_checks,
            )
        command = self._normalize_runtime_command(argv, port=resolved_port)
        return self._execute_test_command(command, timeout_seconds=timeout_seconds)

    def _detect_test_command_name(self) -> str:
        if (self.workspace_root / "manage.py").exists():
            return "django-test"
        return "pytest"

    def _run_runtime_prepare(self, *, timeout_seconds: int) -> dict[str, object]:
        profile = self._detect_runtime_profile()
        if profile is None:
            return TestRunResult(
                success=True,
                command=["runtime-prepare", "skipped"],
                failed_tests=[],
                output_excerpt="no runtime preparation available",
                exit_code=0,
            ).model_dump()
        if profile["kind"] == "django":
            return self._run_django_runtime_prepare(timeout_seconds=timeout_seconds)
        return TestRunResult(
            success=True,
            command=["runtime-prepare", profile["kind"], "skipped"],
            failed_tests=[],
            output_excerpt=f"no runtime preparation required for {profile['kind']}",
            exit_code=0,
        ).model_dump()

    def _run_runtime_smoke(
        self,
        *,
        timeout_seconds: int,
        http_checks: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        profile = self._detect_runtime_profile()
        if profile is None:
            if http_checks:
                return TestRunResult(
                    success=False,
                    command=["runtime-smoke", "missing-profile"],
                    failed_tests=[],
                    output_excerpt=(
                        "runtime HTTP checks were requested, but no supported "
                        "runtime profile was detected"
                    ),
                    exit_code=1,
                ).model_dump()
            return TestRunResult(
                success=True,
                command=["runtime-smoke", "skipped"],
                failed_tests=[],
                output_excerpt="no runtime smoke check available",
                exit_code=0,
            ).model_dump()
        if profile["kind"] == "django":
            return self._run_django_runserver_check(
                timeout_seconds=timeout_seconds,
                http_checks=http_checks,
            )
        if profile["kind"] == "fastapi":
            return self._run_fastapi_runtime_check(
                profile,
                timeout_seconds=timeout_seconds,
                http_checks=http_checks,
            )
        if profile["kind"] == "flask":
            return self._run_flask_runtime_check(
                profile,
                timeout_seconds=timeout_seconds,
                http_checks=http_checks,
            )
        return TestRunResult(
            success=True,
            command=["runtime-smoke", profile["kind"], "skipped"],
            failed_tests=[],
            output_excerpt=f"no runtime smoke check implemented for {profile['kind']}",
            exit_code=0,
        ).model_dump()

    def _run_django_wsgi_check(self, *, timeout_seconds: int) -> dict[str, object]:
        settings_module = self._detect_django_settings_module()
        if settings_module is None:
            raise ValueError("could not detect DJANGO_SETTINGS_MODULE from manage.py")
        command = self._normalize_python_command(
            [
                sys.executable,
                "-c",
                (
                    "import os; "
                    f"os.environ.setdefault('DJANGO_SETTINGS_MODULE', {settings_module!r}); "
                    "from django.core.servers.basehttp import get_internal_wsgi_application; "
                    "get_internal_wsgi_application(); "
                    "print('django runtime smoke ok')"
                ),
            ]
        )
        return self._execute_test_command(command, timeout_seconds=timeout_seconds)

    def _run_django_runtime_prepare(self, *, timeout_seconds: int) -> dict[str, object]:
        steps = [
            self._normalize_python_command([sys.executable, "manage.py", "makemigrations"]),
            self._normalize_python_command([sys.executable, "manage.py", "migrate", "--noinput"]),
        ]
        outputs: list[str] = []
        for command in steps:
            completed = self._run_subprocess(command, timeout_seconds=timeout_seconds)
            output = (completed.stdout + "\n" + completed.stderr).strip()
            outputs.append(f"$ {' '.join(command)}\n{output}".strip())
            if completed.returncode != 0:
                return TestRunResult(
                    success=False,
                    command=command,
                    failed_tests=[],
                    output_excerpt="\n\n".join(outputs)[-20000:],
                    exit_code=completed.returncode,
                ).model_dump()
        return TestRunResult(
            success=True,
            command=steps[-1],
            failed_tests=[],
            output_excerpt="\n\n".join(outputs)[-20000:],
            exit_code=0,
        ).model_dump()

    def _detect_django_settings_module(self) -> str | None:
        manage_path = self.workspace_root / "manage.py"
        if not manage_path.exists():
            return None
        content = manage_path.read_text(encoding="utf-8")
        match = DJANGO_SETTINGS_PATTERN.search(content)
        if match is None:
            return None
        return match.group(1)

    def _run_django_runserver_check(
        self,
        *,
        timeout_seconds: int,
        http_checks: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        port = self._reserve_tcp_port()
        command = self._normalize_python_command(
            [
                sys.executable,
                "manage.py",
                "runserver",
                f"127.0.0.1:{port}",
                "--noreload",
            ]
        )
        return self._run_listening_process_check(
            command,
            port=port,
            timeout_seconds=timeout_seconds,
            http_checks=http_checks,
        )

    def _detect_runtime_profile(self) -> dict[str, str] | None:
        if (self.workspace_root / "manage.py").exists():
            return {"kind": "django"}
        for relative_path in PYTHON_WEB_ENTRYPOINT_CANDIDATES:
            candidate = self.workspace_root / relative_path
            if not candidate.exists() or not candidate.is_file():
                continue
            content = candidate.read_text(encoding="utf-8")
            fastapi_match = FASTAPI_APP_PATTERN.search(content)
            if fastapi_match is not None:
                return {
                    "kind": "fastapi",
                    "module": self._module_path_from_relative_path(relative_path),
                    "attribute": fastapi_match.group(1),
                }
            flask_match = FLASK_APP_PATTERN.search(content)
            if flask_match is not None:
                return {
                    "kind": "flask",
                    "module": self._module_path_from_relative_path(relative_path),
                    "attribute": flask_match.group(1),
                }
        return None

    def _run_fastapi_runtime_check(
        self,
        profile: dict[str, str],
        *,
        timeout_seconds: int,
        http_checks: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self.run_command(
            argv=[
                sys.executable,
                "-m",
                "uvicorn",
                f"{profile['module']}:{profile['attribute']}",
                "--host",
                "{host}",
                "--port",
                "{port}",
            ],
            timeout_seconds=timeout_seconds,
            mode="server",
            http_checks=http_checks,
        )

    def _run_flask_runtime_check(
        self,
        profile: dict[str, str],
        *,
        timeout_seconds: int,
        http_checks: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self.run_command(
            argv=[
                sys.executable,
                "-c",
                (
                    "import importlib; "
                    f"module = importlib.import_module({profile['module']!r}); "
                    f"app = getattr(module, {profile['attribute']!r}); "
                    "app.run(host='{host}', port={port}, use_reloader=False)"
                ),
            ],
            timeout_seconds=timeout_seconds,
            mode="server",
            http_checks=http_checks,
        )

    @staticmethod
    def _module_path_from_relative_path(relative_path: str) -> str:
        path = PurePosixPath(relative_path)
        return ".".join(path.with_suffix("").parts)

    @staticmethod
    def _reserve_tcp_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _is_tcp_port_open(port: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            return False

    def _run_listening_process_check(
        self,
        command: list[str],
        *,
        port: int,
        timeout_seconds: int,
        http_path: str = "/",
        http_checks: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        deadline = time.monotonic() + min(timeout_seconds, MAX_RUNTIME_STARTUP_WAIT_SECONDS)
        process = subprocess.Popen(
            command,
            cwd=self.workspace_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._build_subprocess_env(),
        )
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    stdout, stderr = process.communicate(timeout=1)
                    output = (stdout + "\n" + stderr).strip()
                    return TestRunResult(
                        success=False,
                        command=command,
                        failed_tests=[],
                        output_excerpt=output[-20000:],
                        exit_code=int(process.returncode or 1),
                    ).model_dump()
                if self._is_tcp_port_open(port):
                    probe = (
                        self._run_http_checks(port=port, http_checks=http_checks)
                        if http_checks
                        else self._probe_http_root(port=port, http_path=http_path)
                    )
                    if probe is None:
                        time.sleep(0.1)
                        continue
                    stdout, stderr = self._terminate_process(process)
                    server_output = (stdout + "\n" + stderr).strip()
                    output = "\n\n".join(
                        part
                        for part in (str(probe["output_excerpt"]), server_output)
                        if part
                    ).strip() or "runtime server boot ok"
                    if not bool(probe["success"]):
                        return TestRunResult(
                            success=False,
                            command=command,
                            failed_tests=[],
                            output_excerpt=output[-20000:],
                            exit_code=int(probe["status_code"]),
                        ).model_dump()
                    return TestRunResult(
                        success=True,
                        command=command,
                        failed_tests=[],
                        output_excerpt=output[-20000:],
                        exit_code=0,
                    ).model_dump()
                time.sleep(0.1)
            stdout, stderr = self._terminate_process(process)
            output = ((stdout + "\n" + stderr).strip() + "\nserver did not become ready before timeout").strip()
            return TestRunResult(
                success=False,
                command=command,
                failed_tests=[],
                output_excerpt=output[-20000:],
                exit_code=124,
            ).model_dump()
        finally:
            if process.poll() is None:
                self._terminate_process(process)

    def _run_http_checks(
        self,
        *,
        port: int,
        http_checks: list[dict[str, object]],
    ) -> dict[str, object] | None:
        cookie_jar = CookieJar()
        last_body = ""
        last_url = f"http://127.0.0.1:{port}/"
        outputs: list[str] = []
        for index, raw in enumerate(http_checks, start=1):
            check = RuntimeHttpCheck.model_validate(raw)
            result = self._perform_http_check(
                port=port,
                check=check,
                index=index,
                cookie_jar=cookie_jar,
                last_body=last_body,
                last_url=last_url,
            )
            if result is None:
                return None
            outputs.append(str(result["output_excerpt"]))
            last_body = str(result.get("body", ""))
            last_url = str(result.get("url", last_url))
            if not bool(result["success"]):
                return {
                    "success": False,
                    "status_code": int(result["status_code"]),
                    "output_excerpt": "\n\n".join(outputs)[-20000:],
                }
        return {
            "success": True,
            "status_code": 200,
            "output_excerpt": "\n\n".join(outputs)[-20000:],
        }

    def _perform_http_check(
        self,
        *,
        port: int,
        check: RuntimeHttpCheck,
        index: int,
        cookie_jar: CookieJar,
        last_body: str,
        last_url: str,
    ) -> dict[str, object] | None:
        opener = self._build_http_opener(cookie_jar=cookie_jar, follow_redirects=check.follow_redirects)
        url = f"http://127.0.0.1:{port}{check.path}"
        headers = dict(check.headers)
        data: bytes | None = None
        if check.form is not None:
            form_payload = {key: str(value) for key, value in check.form.items()}
            if check.use_csrf_from_last_response and check.method in {"POST", "PUT", "PATCH", "DELETE"}:
                csrf_form_token = self._extract_csrf_token(last_body) or self._cookie_value(
                    cookie_jar,
                    "csrftoken",
                )
                if csrf_form_token is not None and "csrfmiddlewaretoken" not in form_payload:
                    form_payload["csrfmiddlewaretoken"] = csrf_form_token
                csrf_cookie = self._cookie_value(cookie_jar, "csrftoken")
                if csrf_cookie is not None and "X-CSRFToken" not in headers:
                    headers["X-CSRFToken"] = csrf_cookie
                headers.setdefault("Referer", last_url)
            data = urllib.parse.urlencode(form_payload).encode("utf-8")
            headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        elif check.json_payload is not None:
            data = json.dumps(check.json_payload).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        elif check.body is not None:
            data = check.body.encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=check.method)
        try:
            with opener.open(request, timeout=0.5) as response:
                status_code = int(getattr(response, "status", response.getcode()))
                body = response.read(4000).decode("utf-8", errors="replace")
                final_url = response.geturl()
        except urllib.error.HTTPError as exc:
            status_code = int(exc.code)
            body = exc.read(4000).decode("utf-8", errors="replace")
            final_url = exc.geturl()
        except (urllib.error.URLError, TimeoutError, OSError):
            return None
        outputs = [
            f"{check.name or f'check {index}'}: {check.method} {check.path} -> HTTP {status_code}"
        ]
        success = status_code == check.expect_status
        if not success:
            outputs.append(f"expected status {check.expect_status}")
        for token in check.body_contains:
            if token not in body:
                success = False
                outputs.append(f"missing body text: {token}")
        for token in check.body_not_contains:
            if token in body:
                success = False
                outputs.append(f"unexpected body text: {token}")
        if body.strip():
            outputs.append(body.strip()[:1000])
        return {
            "success": success,
            "status_code": status_code,
            "output_excerpt": "\n".join(outputs)[-20000:],
            "body": body,
            "url": final_url,
        }

    @staticmethod
    def _build_http_opener(*, cookie_jar: CookieJar, follow_redirects: bool):
        handlers: list[object] = [urllib.request.HTTPCookieProcessor(cookie_jar)]
        if not follow_redirects:
            class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    return None

            handlers.append(_NoRedirectHandler())
        return urllib.request.build_opener(*handlers)

    @staticmethod
    def _cookie_value(cookie_jar: CookieJar, name: str) -> str | None:
        for cookie in cookie_jar:
            if cookie.name == name:
                return cookie.value
        return None

    @staticmethod
    def _extract_csrf_token(body: str) -> str | None:
        match = CSRF_INPUT_PATTERN.search(body)
        if match is None:
            return None
        return match.group(1) or match.group(2)

    def _probe_http_root(self, *, port: int, http_path: str = "/") -> dict[str, object] | None:
        url = f"http://127.0.0.1:{port}{http_path}"
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                status_code = int(getattr(response, "status", response.getcode()))
                body = response.read(4000).decode("utf-8", errors="replace").strip()
        except urllib.error.HTTPError as exc:
            status_code = int(exc.code)
            body = exc.read(4000).decode("utf-8", errors="replace").strip()
        except (urllib.error.URLError, TimeoutError, OSError):
            return None
        output = f"GET {http_path} -> HTTP {status_code}"
        if body:
            output = f"{output}\n{body}"
        return {
            "success": status_code < 500,
            "status_code": status_code,
            "output_excerpt": output[-20000:],
        }

    @staticmethod
    def _normalize_runtime_command(argv: list[str], *, port: int | None = None, host: str = "127.0.0.1") -> list[str]:
        resolved = [
            item.replace("{host}", host).replace("{port}", str(port) if port is not None else "{port}")
            for item in argv
        ]
        return TestServer._normalize_python_command(resolved)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> tuple[str, str]:
        if process.poll() is None:
            process.terminate()
            try:
                return process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        return process.communicate()

    @staticmethod
    def _normalize_python_command(command: list[str]) -> list[str]:
        if not command:
            return command
        first = command[0]
        try:
            is_python = Path(first).resolve() == Path(sys.executable).resolve()
        except OSError:
            is_python = first == sys.executable
        if is_python:
            executable = _PythonExecutable(sys.executable)
            if len(command) >= 2 and command[1] == "-B":
                return [executable, *command[1:]]
            return [executable, "-B", *command[1:]]
        return command

    def _build_subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(self.workspace_root) if not existing else f"{self.workspace_root}{os.pathsep}{existing}"
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        return env

    def _run_subprocess(self, command: list[str], *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        for cache_dir in self.workspace_root.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
        return subprocess.run(
            command,
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=self._build_subprocess_env(),
            check=False,
        )

    def _execute_test_command(self, command: list[str], *, timeout_seconds: int) -> dict[str, object]:
        completed = self._run_subprocess(command, timeout_seconds=timeout_seconds)
        output = (completed.stdout + "\n" + completed.stderr).strip()
        failed_tests = [
            line.strip()
            for line in output.splitlines()
            if "::" in line and ("FAILED" in line or "ERROR" in line)
        ]
        payload = TestRunResult(
            success=completed.returncode == 0,
            command=command,
            failed_tests=failed_tests,
            output_excerpt=output[-20000:],
            exit_code=completed.returncode,
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

    @staticmethod
    def _is_virtualenv_python() -> bool:
        return sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))


TestServer.__test__ = False


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
        self.approval_notifications: list[dict[str, object]] = []
        self.runtime_notifications: list[dict[str, object]] = []

    def send_notification(
        self,
        body: str | None = None,
        message: str | None = None,
        **metadata: object,
    ) -> dict[str, object]:
        text = redact_text((body if body is not None else message or "")[:2000])
        self.notifications.append(text)
        payload = {"message": text, "channel": "console", **metadata}
        kind = str(metadata.get("kind", ""))
        if kind == "approval_required" or metadata.get("approval_id") is not None:
            self.approval_notifications.append(payload)
        if kind.startswith("runtime") or metadata.get("runtime_issue_id") is not None:
            self.runtime_notifications.append(payload)
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
        router.register("test_server.run_command", self.test_server.run_command)
        router.register("test_server.install_package", self.test_server.install_package)
        router.register("memory_server.write_memory", self.memory_server.write_memory)
        router.register("memory_server.read_memory", self.memory_server.read_memory)
        router.register("memory_server.search_memory", self.memory_server.search_memory)
        router.register("memory_server.update_task_summary", self.memory_server.update_task_summary)
        router.register("notify_server.send_notification", self.notify_server.send_notification)
        return router
