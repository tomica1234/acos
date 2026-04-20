from packages.schemas.tasks import PlannedTask

from packages.orchestrator.context_builder import ContextBuilder


def test_context_builder_truncates_large_sections() -> None:
    builder = ContextBuilder()
    packet = builder.build(
        job_id="job-1",
        role="architect",
        objective="Design the system",
        repo_path=".",
        request_text="api_key=sk-abcdefghijklmnopqrstuvwxyz" + ("x" * 4000),
        constraints=["no secrets"],
        relevant_files={"a.py": "print('a')\n" * 200, "b.py": "print('b')\n" * 200},
        diff="+ very long diff\n" * 200,
        memory_summaries=["memory\n" * 200],
        logs=["log\n" * 200],
        token_budget=256,
    )

    rendered = packet.render_text()
    assert "[truncated]" in rendered
    assert len(rendered) < 3000
    assert "[REDACTED]" in rendered


def test_context_builder_redacts_task_fields() -> None:
    builder = ContextBuilder()
    packet = builder.build(
        job_id="job-2",
        role="fixer",
        objective="Fix the failure",
        repo_path=".",
        request_text="normal request",
        constraints=[],
        relevant_files={},
        diff="",
        memory_summaries=[],
        logs=[],
        token_budget=256,
        task=PlannedTask(
            id="task-1",
            title="Investigate api_key=sk-abcdefghijklmnopqrstuvwxyz",
            description="The password=super-secret value leaked",
            role="fixer",
        ),
    )

    rendered = packet.render_text()
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in rendered
    assert "super-secret" not in rendered
    assert "[REDACTED]" in rendered
