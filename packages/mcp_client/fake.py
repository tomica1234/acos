"""Fake MCP servers used for tests and local demos."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable

from packages.memory.redaction import redact_text
from packages.memory.store import SQLiteMemoryStore
from packages.mcp_client.router import MCPRouter
from packages.orchestrator.workspace import WorkspacePolicy
from packages.schemas.agent_outputs import TestRunResult
from packages.schemas.runtime import RuntimeHttpCheck

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

TEST_COMMAND_ALLOWLIST: dict[str, list[str]] = {
    "python-compile": [sys.executable, "-m", "compileall", "."],
    "pytest": [sys.executable, "-m", "pytest", "-q"],
    "pytest-unit": [sys.executable, "-m", "pytest", "tests", "-q"],
    "django-test": [sys.executable, "manage.py", "test"],
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
            return self._run_runtime_smoke(timeout_seconds=timeout_seconds, http_checks=http_checks)
        if command_name == "django-wsgi-check":
            return self._run_django_wsgi_check(timeout_seconds=timeout_seconds)
        if command_name not in TEST_COMMAND_ALLOWLIST:
            raise ValueError(f"command_name {command_name} is not allowlisted")
        if timeout_seconds <= 0 or timeout_seconds > MAX_TEST_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be between 1 and {MAX_TEST_TIMEOUT_SECONDS}"
            )
        command = self._normalize_python_command(list(TEST_COMMAND_ALLOWLIST[command_name]))
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
        output = f"GET / -> HTTP {status_code}"
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
        if len(command) >= 3 and command[0] == sys.executable and command[1] != "-B":
            return [command[0], "-B", *command[1:]]
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
        )
        return payload.model_dump()

    @staticmethod
    def _is_virtualenv_python() -> bool:
        return sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))


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
        router.register("test_server.run_command", self.test_server.run_command)
        router.register("test_server.install_package", self.test_server.install_package)
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
