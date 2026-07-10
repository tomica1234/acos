import importlib.metadata
import subprocess
import sys
from pathlib import Path

from packages.mcp_client import fake
from packages.mcp_client.fake import TestServer
from packages.schemas.agent_outputs import TestRunResult


def test_test_server_installs_missing_generated_requirements(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "requirements.txt").write_text(
        "aiosqlite\npytest-asyncio\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_smoke.py").write_text(
        "def test_smoke():\n    assert True\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_version(package_name: str) -> str:
        if package_name in {"aiosqlite", "pytest-asyncio"}:
            raise importlib.metadata.PackageNotFoundError(package_name)
        return "1.0.0"

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[:3] == [sys.executable, "-m", "pip"]:
            return subprocess.CompletedProcess(command, 0, stdout="installed", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="1 passed in 0.01s",
            stderr="",
        )

    monkeypatch.setattr(fake.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(fake.subprocess, "run", fake_run)

    result = TestRunResult.model_validate(TestServer(tmp_path).run_test())

    assert result.success is True
    assert calls[0] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "aiosqlite",
        "pytest-asyncio",
    ]
    assert calls[1][:3] == [sys.executable, "-B", "-m"]


def test_test_server_reports_dependency_install_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "requirements.txt").write_text("missing-package\n", encoding="utf-8")

    def fake_version(package_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(package_name)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="not found")

    monkeypatch.setattr(fake.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(fake.subprocess, "run", fake_run)

    result = TestRunResult.model_validate(TestServer(tmp_path).run_test())

    assert result.success is False
    assert result.command[:3] == [sys.executable, "-m", "pip"]
    assert "Dependency installation failed before tests could run" in result.output_excerpt


def test_test_server_installs_inferred_playwright_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_project_init.py").write_text(
        "import subprocess\n"
        "import sys\n\n"
        "def test_playwright_installed():\n"
        "    subprocess.run([sys.executable, \"-m\", \"playwright\", \"--version\"])\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_version(package_name: str) -> str:
        if package_name == "playwright":
            raise importlib.metadata.PackageNotFoundError(package_name)
        return "1.0.0"

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="1 passed", stderr="")

    monkeypatch.setattr(fake.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(fake.subprocess, "run", fake_run)

    result = TestRunResult.model_validate(TestServer(tmp_path).run_test())

    assert result.success is True
    assert calls[0] == [sys.executable, "-m", "pip", "install", "playwright"]
    assert calls[1][:3] == [sys.executable, "-B", "-m"]


def test_test_server_installs_inferred_async_python_dependencies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "database.py").write_text(
        'DATABASE_URL = "sqlite+aiosqlite:///./test.db"\n',
        encoding="utf-8",
    )
    tests_dir = backend / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_database.py").write_text(
        "import pytest\n"
        "import pytest_asyncio\n\n"
        "@pytest.mark.asyncio\n"
        "async def test_smoke():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_version(package_name: str) -> str:
        if package_name in {"aiosqlite", "pytest-asyncio"}:
            raise importlib.metadata.PackageNotFoundError(package_name)
        return "1.0.0"

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="1 passed", stderr="")

    monkeypatch.setattr(fake.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(fake.subprocess, "run", fake_run)

    result = TestRunResult.model_validate(TestServer(tmp_path).run_test())

    assert result.success is True
    assert calls[0] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "aiosqlite",
        "pytest-asyncio",
    ]
    assert calls[1][:3] == [sys.executable, "-B", "-m"]
