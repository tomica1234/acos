from __future__ import annotations

import io
import urllib.parse
from http.cookiejar import Cookie, CookieJar
from pathlib import Path
import urllib.error

import pytest

from packages.agents.runner import AgentRunner
from packages.llm.routing import RoutingContext
from packages.mcp_client.fake import FakeMCPEnvironment, RepoServer, TestServer
from packages.mcp_client.router import MCPRouter
from packages.memory.redaction import redact_text
from packages.orchestrator.audit import AuditRecorder
from packages.orchestrator.job_runner import JobRunner
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.runtime import RuntimeHttpCheck
from packages.schemas.agent_outputs import FilePatch, ImplementationResult, TestRunResult
from packages.schemas.context import ContextPacket
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import ImplementationStatus

from tests.conftest import attach_mock_adapter, config_dir, load_registry


def _packet(role: str, repo_path: str = ".") -> ContextPacket:
    return ContextPacket(
        job_id=f"job-{role}",
        role=role,
        objective=f"Run {role}",
        repo_path=repo_path,
        request_text="Use secret=sk-this-should-be-redacted",
        token_budget=4096,
        model_context_budget=4096,
        selected_model_hint="auto",
    )


def test_redaction_covers_common_secret_formats() -> None:
    text = "\n".join(
        [
            "api_key=sk-abcdefghijklmnopqrstuvwxyz",
            "Authorization: Bearer token-value-1234567890",
            "aws_access_key_id=AKIAABCDEFGHIJKLMNOP",
            "password='super-secret'",
            "-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----",
        ]
    )

    redacted = redact_text(text)

    assert "sk-abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "super-secret" not in redacted
    assert "PRIVATE KEY" not in redacted
    assert "[REDACTED]" in redacted


def test_repo_server_blocks_secret_paths_and_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (workspace / "inside.py").write_text("VALUE = 1\n", encoding="utf-8")
    try:
        (workspace / "link.txt").symlink_to(outside_file)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable in this environment: {exc}")
    repo = RepoServer(workspace)

    assert repo.repo_tree()["files"] == ["inside.py"]

    with pytest.raises(ValueError):
        repo.read_file(".env.local")

    with pytest.raises(ValueError):
        repo.read_file(".git/config")

    with pytest.raises(ValueError):
        repo.apply_patch("config/secret_key", "value = 1\n", operation="create")

    with pytest.raises(ValueError):
        repo.read_file("link.txt")


def test_test_server_rejects_invalid_timeout_and_unknown_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = TestServer(workspace)

    with pytest.raises(ValueError):
        server.run_test(command_name="pytest", timeout_seconds=0)

    with pytest.raises(ValueError):
        server.run_test(command_name="python -c bad", timeout_seconds=30)


def test_test_server_auto_selects_django_manage_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "manage.py").write_text("print('placeholder')\n", encoding="utf-8")
    server = TestServer(workspace)
    captured: dict[str, object] = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        return Result()

    monkeypatch.setattr("packages.mcp_client.fake.subprocess.run", fake_run)

    payload = server.run_test()

    assert payload["success"] is True
    assert captured["command"][0] == Path(__import__("sys").executable).as_posix()
    assert captured["command"][1:] == ["-B", "manage.py", "test"]


def test_test_server_runtime_smoke_starts_django_runserver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "manage.py").write_text(
        "\n".join(
            [
                "import os",
                "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mytodo.settings')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    server = TestServer(workspace)
    captured: dict[str, object] = {}

    def fake_boot(self, command, *, port, timeout_seconds, http_path="/", http_checks=None):
        captured["command"] = command
        captured["port"] = port
        captured["timeout_seconds"] = timeout_seconds
        captured["http_path"] = http_path
        captured["http_checks"] = http_checks
        return TestRunResult(
            success=True,
            command=command,
            failed_tests=[],
            output_excerpt="runtime server boot ok",
            exit_code=0,
        ).model_dump()

    monkeypatch.setattr(TestServer, "_reserve_tcp_port", staticmethod(lambda: 8765))
    monkeypatch.setattr(TestServer, "_run_listening_process_check", fake_boot)

    payload = server.run_test(command_name="runtime-smoke-auto")

    assert payload["success"] is True
    assert captured["command"][1:4] == ["-B", "manage.py", "runserver"]
    assert captured["command"][4:] == ["127.0.0.1:8765", "--noreload"]
    assert captured["port"] == 8765


def test_test_server_runtime_prepare_runs_django_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "manage.py").write_text(
        "\n".join(
            [
                "import os",
                "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mytodo.settings')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    server = TestServer(workspace)
    captured: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    def fake_run(command, **kwargs):
        captured.append(command)
        return Result()

    monkeypatch.setattr("packages.mcp_client.fake.subprocess.run", fake_run)

    payload = server.run_test(command_name="prepare-runtime-auto")

    assert payload["success"] is True
    assert len(captured) == 2
    assert captured[0][1:] == ["-B", "manage.py", "makemigrations"]
    assert captured[1][1:] == ["-B", "manage.py", "migrate", "--noinput"]


def test_test_server_runtime_smoke_detects_fastapi_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app").mkdir()
    (workspace / "app" / "main.py").write_text(
        "\n".join(
            [
                "from fastapi import FastAPI",
                "",
                "app = FastAPI()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    server = TestServer(workspace)
    captured: dict[str, object] = {}

    def fake_boot(self, command, *, port, timeout_seconds, http_path="/", http_checks=None):
        captured["command"] = command
        captured["port"] = port
        captured["timeout_seconds"] = timeout_seconds
        captured["http_path"] = http_path
        captured["http_checks"] = http_checks
        return TestRunResult(
            success=True,
            command=command,
            failed_tests=[],
            output_excerpt="runtime server boot ok",
            exit_code=0,
        ).model_dump()

    monkeypatch.setattr(TestServer, "_reserve_tcp_port", staticmethod(lambda: 8123))
    monkeypatch.setattr(TestServer, "_run_listening_process_check", fake_boot)

    payload = server.run_test(command_name="runtime-smoke-auto")

    assert payload["success"] is True
    assert captured["command"][1:4] == ["-B", "-m", "uvicorn"]
    assert "app.main:app" in captured["command"]
    assert captured["command"][-4:] == ["--host", "127.0.0.1", "--port", "8123"]
    assert captured["port"] == 8123
    assert captured["http_path"] == "/"


def test_test_server_run_command_formats_server_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = TestServer(workspace)
    captured: dict[str, object] = {}

    def fake_boot(self, command, *, port, timeout_seconds, http_path="/", http_checks=None):
        captured["command"] = command
        captured["port"] = port
        captured["timeout_seconds"] = timeout_seconds
        captured["http_path"] = http_path
        captured["http_checks"] = http_checks
        return TestRunResult(
            success=True,
            command=command,
            failed_tests=[],
            output_excerpt="runtime server boot ok",
            exit_code=0,
        ).model_dump()

    monkeypatch.setattr(TestServer, "_reserve_tcp_port", staticmethod(lambda: 9234))
    monkeypatch.setattr(TestServer, "_run_listening_process_check", fake_boot)

    payload = server.run_command(
        argv=[__import__("sys").executable, "manage.py", "runserver", "{host}:{port}", "--noreload"],
        mode="server",
        http_path="/healthz",
    )

    assert payload["success"] is True
    assert captured["command"][1:4] == ["-B", "manage.py", "runserver"]
    assert captured["command"][4:] == ["127.0.0.1:9234", "--noreload"]
    assert captured["port"] == 9234
    assert captured["http_path"] == "/healthz"


def test_test_server_http_checks_support_browser_like_form_crud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = TestServer(workspace)
    cookie_jar = CookieJar()
    cookie_jar.set_cookie(
        Cookie(
            version=0,
            name="csrftoken",
            value="cookie-secret",
            port=None,
            port_specified=False,
            domain="127.0.0.1",
            domain_specified=False,
            domain_initial_dot=False,
            path="/",
            path_specified=True,
            secure=False,
            expires=None,
            discard=True,
            comment=None,
            comment_url=None,
            rest={},
            rfc2109=False,
        )
    )

    class FakeResponse:
        def __init__(self, *, url: str, body: str, status: int = 200) -> None:
            self.status = status
            self._url = url
            self._body = body.encode("utf-8")

        def read(self, size: int = -1) -> bytes:
            return self._body if size < 0 else self._body[:size]

        def getcode(self) -> int:
            return self.status

        def geturl(self) -> str:
            return self._url

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeOpener:
        def open(self, request, timeout: float = 0.5):
            payload = urllib.parse.parse_qs((request.data or b"").decode("utf-8"))
            header_value = (
                request.headers.get("X-CSRFToken")
                or request.headers.get("X-Csrftoken")
                or request.headers.get("X-csrftoken")
            )
            assert header_value == "cookie-secret"
            assert payload["csrfmiddlewaretoken"][0] in {"form-token", "cookie-secret"}
            assert payload["title"][0] == "milk"
            return FakeResponse(
                url=request.full_url,
                body="<html><body><ul><li>milk</li></ul></body></html>",
            )

    monkeypatch.setattr(TestServer, "_build_http_opener", staticmethod(lambda **kwargs: FakeOpener()))

    create_result = server._perform_http_check(
        port=8000,
        check=RuntimeHttpCheck.model_validate(
            {
                "name": "create",
                "method": "POST",
                "path": "/create/",
                "form": {"title": "milk"},
                "expect_status": 200,
                "body_contains": ["milk"],
            }
        ),
        index=1,
        cookie_jar=cookie_jar,
        last_body=(
            "<html><body>"
            "<input type='hidden' name='csrfmiddlewaretoken' value='form-token'>"
            "</body></html>"
        ),
        last_url="http://127.0.0.1:8000/",
    )

    delete_result = {
        "success": True,
        "status_code": 200,
        "output_excerpt": "delete: GET /delete/1 -> HTTP 200",
        "body": "<html><body><ul></ul></body></html>",
        "url": "http://127.0.0.1:8000/delete/1",
    }
    monkeypatch.setattr(
        TestServer,
        "_perform_http_check",
        lambda self, **kwargs: create_result if kwargs["index"] == 1 else delete_result,
    )

    payload = server._run_http_checks(
        port=8000,
        http_checks=[
            {
                "name": "create",
                "method": "POST",
                "path": "/create/",
                "form": {"title": "milk"},
                "expect_status": 200,
                "body_contains": ["milk"],
            },
            {
                "name": "delete",
                "path": "/delete/1",
                "expect_status": 200,
                "body_not_contains": ["milk"],
            },
        ],
    )

    assert create_result is not None
    assert create_result["success"] is True
    assert payload is not None
    assert payload["success"] is True
    assert "create: POST /create/ -> HTTP 200" in str(payload["output_excerpt"])
    assert "delete: GET /delete/1 -> HTTP 200" in str(payload["output_excerpt"])


def test_test_server_http_probe_flags_runtime_500(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = TestServer(workspace)

    def fake_urlopen(url: str, timeout: float = 0.5):
        raise urllib.error.HTTPError(
            url=url,
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b"no such table: todos_todo"),
        )

    monkeypatch.setattr("packages.mcp_client.fake.urllib.request.urlopen", fake_urlopen)

    payload = server._probe_http_root(port=8000)

    assert payload is not None
    assert payload["success"] is False
    assert payload["status_code"] == 500
    assert "no such table: todos_todo" in str(payload["output_excerpt"])


def test_test_server_install_package_uses_virtualenv_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = TestServer(workspace)
    captured: dict[str, object] = {}

    class Result:
        returncode = 0
        stdout = "installed\n"
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        return Result()

    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / ".venv"))
    monkeypatch.setattr("packages.mcp_client.fake.subprocess.run", fake_run)

    payload = server.install_package("django>=5,<6")

    assert payload["success"] is True
    assert captured["command"][0] == Path(__import__("sys").executable).as_posix()
    assert captured["command"][1:] == ["-B", "-m", "pip", "install", "django>=5,<6"]


def test_policy_blocks_test_and_dependency_patches() -> None:
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    with pytest.raises(PermissionError):
        policy.assert_patch_target_allowed("implementer", "tests/test_feature.py")

    with pytest.raises(PermissionError):
        policy.assert_patch_target_allowed("fixer", "tests/test_feature.py")

    with pytest.raises(PermissionError):
        policy.assert_patch_target_allowed("implementer", "pyproject.toml")

    with pytest.raises(PermissionError):
        policy.assert_patch_target_allowed("test_writer", ".env.local")

    policy.assert_patch_target_allowed("test_writer", "tests/test_feature.py")


def test_agent_runner_rejects_tool_override_outside_agent_config(tmp_path: Path) -> None:
    registry = load_registry()
    registry.agents["implementer"].allowed_tools = ["repo_server.read_file"]
    attach_mock_adapter(
        registry,
        {
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="unused",
                patches=[],
            ).model_dump()
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=tmp_path,
        memory_db_path=tmp_path / ".memory.sqlite3",
    )
    runner = AgentRunner(
        registry=registry,
        mcp_router=environment.build_router(),
        policy_engine=policy,
        audit_recorder=AuditRecorder(),
    )

    with pytest.raises(PermissionError):
        runner.run(
            role="implementer",
            response_model=ImplementationResult,
            context_packet=_packet("implementer", repo_path=str(tmp_path)),
            routing_context=RoutingContext(role="implementer"),
            allowed_tools=["repo_server.search_text"],
        )


def test_agent_runner_blocks_forbidden_patch_tool_call(tmp_path: Path) -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "implementer": {
                "content": "",
                "tool_calls": [
                    {
                        "name": "repo_server.apply_patch",
                        "arguments": {
                            "path": "tests/test_feature.py",
                            "content": "def test_bad() -> None:\n    assert True\n",
                            "operation": "create",
                        },
                    }
                ],
            }
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=tmp_path,
        memory_db_path=tmp_path / ".memory.sqlite3",
    )
    runner = AgentRunner(
        registry=registry,
        mcp_router=environment.build_router(),
        policy_engine=policy,
        audit_recorder=AuditRecorder(),
    )

    with pytest.raises(PermissionError):
        runner.run(
            role="implementer",
            response_model=ImplementationResult,
            context_packet=_packet("implementer", repo_path=str(tmp_path)),
            routing_context=RoutingContext(role="implementer"),
        )


def test_job_runner_blocks_dependency_manifest_patch(tmp_path: Path) -> None:
    registry = load_registry()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=tmp_path,
        memory_db_path=tmp_path / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = JobRecord(
        job_id="job-1",
        spec=JobSpec(
            request_text="test",
            repo_path=str(tmp_path),
            target_branch="acos/security-check",
        ),
    )

    with pytest.raises(PermissionError):
        runner._apply_patches(
            record,
            "implementer",
            [FilePatch(path="pyproject.toml", content="[project]\nname='bad'\n")],
        )


def test_memory_and_notify_servers_redact_secret_payloads(tmp_path: Path) -> None:
    environment = FakeMCPEnvironment(
        workspace_root=tmp_path,
        memory_db_path=tmp_path / "memory.sqlite3",
    )

    environment.memory_server.write_memory(
        uri="memory://job-1/secret",
        content="token=sk-abcdefghijklmnopqrstuvwxyz",
    )
    entries = environment.memory_server.read_memory(uri="memory://job-1")
    serialized = str(entries)
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "[REDACTED]" in serialized

    payload = environment.notify_server.send_notification(
        body="Authorization: Bearer abcdefghijklmnopqrstuvwxyz"
    )
    assert "abcdefghijklmnopqrstuvwxyz" not in str(payload)
    assert "[REDACTED]" in str(payload)


def test_mcp_router_redacts_tool_errors() -> None:
    router = MCPRouter()

    def bad_handler() -> dict[str, str]:
        raise RuntimeError("api_key=sk-abcdefghijklmnopqrstuvwxyz")

    router.register("demo.bad", bad_handler)
    result = router.call("demo.bad")

    assert not result.ok
    assert result.error is not None
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in result.error
    assert "[REDACTED]" in result.error
