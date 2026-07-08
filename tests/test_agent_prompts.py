from packages.agents import planner


def test_planner_prompt_requires_task_artifact_paths() -> None:
    prompt = planner.SYSTEM_PROMPT

    assert "target_files" in prompt
    assert "required_artifacts" in prompt
    assert '"role": "scaffold|implementer|test_writer"' in prompt
    assert "implementer, scaffold, and test_writer" in prompt
    assert "do not schedule pm, architect, reviewer" in prompt
    assert "exact test file path" in prompt
    assert "repo-relative file path" in prompt
