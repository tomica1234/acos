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

