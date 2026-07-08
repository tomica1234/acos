from __future__ import annotations

from packages.orchestrator.execution_contracts import synthesize_job_metadata_from_prd
from packages.schemas.agent_outputs import PRD, RuntimePlan
from packages.schemas.runtime import RuntimeHttpCheck


def test_synthesize_job_metadata_from_prd_uses_explicit_fastapi_contract(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prd = PRD(
        title="Status API",
        problem_statement="Build a FastAPI status API.",
        framework_profile="fastapi-api",
        framework_entrypoint="app.main:app",
        required_artifacts=["tests/test_app.py"],
        acceptance_checks=[
            RuntimeHttpCheck(
                name="home",
                method="GET",
                path="/",
                expect_status=200,
            )
        ],
        runtime=RuntimePlan(
            http_checks=[
                RuntimeHttpCheck(
                    name="health",
                    method="GET",
                    path="/healthz",
                    expect_status=200,
                )
            ]
        ),
    )

    metadata = synthesize_job_metadata_from_prd(
        prd,
        {},
        workspace_root=workspace,
    )

    assert metadata["framework_profile"] == "fastapi-api"
    assert metadata["framework_entrypoint"] == "app.main:app"
    assert metadata["required_artifacts"] == [
        "app/__init__.py",
        "app/main.py",
        "tests/test_app.py",
    ]
    assert metadata["runtime"]["start_command"] == [
        "python",
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "{host}",
        "--port",
        "{port}",
    ]
    assert metadata["runtime"]["http_checks"][0]["path"] == "/healthz"
    assert metadata["acceptance_checks"][0]["path"] == "/"


def test_prd_runtime_unknown_hints_are_preserved_as_extra(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prd = PRD.model_validate(
        {
            "title": "English Vocabulary App",
            "problem_statement": "Build a vocabulary learning app.",
            "runtime": {
                "python": {"version": "3.11", "backend": "FastAPI"},
                "node": {"version": "22", "frontend": "Vite"},
                "http_probe_path": "/health",
            },
        }
    )

    metadata = synthesize_job_metadata_from_prd(
        prd,
        {},
        workspace_root=workspace,
    )

    assert prd.runtime is not None
    assert prd.runtime.extra["python"] == {"version": "3.11", "backend": "FastAPI"}
    assert prd.runtime.extra["node"] == {"version": "22", "frontend": "Vite"}
    assert metadata["runtime"]["python"] == {"version": "3.11", "backend": "FastAPI"}
    assert metadata["runtime"]["node"] == {"version": "22", "frontend": "Vite"}
    assert metadata["runtime"]["http_probe_path"] == "/health"


def test_prd_runtime_normalizes_string_commands() -> None:
    prd = PRD.model_validate(
        {
            "title": "English Vocabulary App",
            "problem_statement": "Build a vocabulary learning app.",
            "runtime": {
                "prepare_commands": [
                    "npm install",
                    ["python", "-m", "pytest"],
                    "",
                ],
                "start_command": "npm run dev -- --host 127.0.0.1 --port {port}",
            },
        }
    )

    assert prd.runtime is not None
    assert prd.runtime.prepare_commands == [
        ["npm", "install"],
        ["python", "-m", "pytest"],
    ]
    assert prd.runtime.start_command == [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        "{port}",
    ]


def test_prd_runtime_normalizes_single_prepare_command_string() -> None:
    prd = PRD.model_validate(
        {
            "title": "API",
            "problem_statement": "Build an API.",
            "runtime": {
                "prepare_commands": "python -m pytest",
            },
        }
    )

    assert prd.runtime is not None
    assert prd.runtime.prepare_commands == [["python", "-m", "pytest"]]


def test_synthesize_job_metadata_filters_invalid_required_artifacts(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prd = PRD(
        title="Feature",
        problem_statement="Build a feature.",
        required_artifacts=[
            "src/feature.py",
            "../outside.py",
            "C:\\outside.py",
            "docs/",
        ],
    )

    metadata = synthesize_job_metadata_from_prd(
        prd,
        {"required_artifacts": ["tests/test_feature.py", "/absolute.py"]},
        workspace_root=workspace,
    )

    assert metadata["required_artifacts"] == [
        "src/feature.py",
        "tests/test_feature.py",
    ]


def test_synthesize_job_metadata_from_prd_inferrs_django_contract(tmp_path) -> None:
    workspace = tmp_path / "my-product"
    workspace.mkdir()
    prd = PRD(
        title="Todo Project",
        problem_statement="Build a Django todo web app with SQLite.",
        goals=["Use Django and template rendering."],
        success_criteria=["README should describe setup and run commands."],
    )

    metadata = synthesize_job_metadata_from_prd(
        prd,
        {},
        workspace_root=workspace,
    )

    assert metadata["framework_profile"] == "django-web"
    assert metadata["framework_project_name"] == "Todo_Project"
    assert "manage.py" in metadata["required_artifacts"]
    assert "README.md" in metadata["required_artifacts"]
    assert any(path.endswith("/settings.py") for path in metadata["required_artifacts"])
    assert metadata["runtime"]["prepare_commands"][0] == ["python", "manage.py", "makemigrations"]
    assert metadata["acceptance_checks"][0]["path"] == "/"
