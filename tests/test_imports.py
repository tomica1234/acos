from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from apps.api.main import create_app
from apps.cli import build_demo_runner
from packages.agents.runner import AgentRunner
from packages.llm.client import LLMClient
from packages.mcp_client.router import MCPRouter
from packages.orchestrator.job_runner import JobRunner


def test_imports() -> None:
    app = create_app
    assert callable(app)
    assert callable(build_demo_runner)
    assert AgentRunner is not None
    assert LLMClient is not None
    assert MCPRouter is not None
    assert JobRunner is not None


def test_cli_module_entrypoint_runs_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "apps.cli", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage: acos" in result.stdout


def test_mcp_and_policy_modules_import_in_fresh_interpreter() -> None:
    modules = [
        "packages.mcp_client.fake",
        "packages.orchestrator.policy",
        "mcp_servers.repo_server.main",
        "mcp_servers.git_server.main",
        "mcp_servers.test_server.main",
    ]
    code = "import importlib\n" + "\n".join(
        f"importlib.import_module({module!r})" for module in modules
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

