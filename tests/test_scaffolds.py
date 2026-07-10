from packages.mcp_client.fake import FakeMCPEnvironment
from packages.orchestrator.scaffolds import build_django_todo_scaffold
from packages.schemas.agent_outputs import TestRunResult


def test_django_todo_scaffold_runs_django_tests(tmp_path) -> None:
    implementation, test_writer = build_django_todo_scaffold()
    environment = FakeMCPEnvironment(
        workspace_root=tmp_path,
        memory_db_path=tmp_path / ".memory.sqlite3",
    )

    for patch in [*implementation.patches, *test_writer.patches]:
        environment.repo_server.apply_patch(
            path=patch.path,
            content=patch.content,
            operation=patch.operation,
        )
    result = TestRunResult.model_validate(
        environment.test_server.run_test(command_name="django-test")
    )

    assert result.success is True
    assert result.executed_test_count is not None
    assert result.executed_test_count >= 1
    assert (tmp_path / "manage.py").exists()
    assert (tmp_path / "todos" / "models.py").exists()
    assert (tmp_path / "todos" / "tests.py").exists()
