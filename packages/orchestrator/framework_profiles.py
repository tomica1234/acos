"""Deterministic framework profile defaults for ACOS jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ResolvedFrameworkProfile:
    key: str
    required_artifacts: tuple[str, ...]
    runtime_prepare_commands: tuple[tuple[str, ...], ...] | None = None
    runtime_start_command: tuple[str, ...] | None = None
    runtime_http_probe_path: str = "/"


def resolve_framework_profile(metadata: Mapping[str, Any]) -> ResolvedFrameworkProfile | None:
    raw_name = metadata.get("framework_profile") or metadata.get("framework")
    if raw_name is None:
        return None
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("metadata.framework_profile must be a non-empty string")
    name = raw_name.strip()
    if name == "django-web":
        return ResolvedFrameworkProfile(
            key=name,
            required_artifacts=("manage.py",),
            runtime_prepare_commands=(
                ("python", "manage.py", "makemigrations"),
                ("python", "manage.py", "migrate", "--noinput"),
            ),
            runtime_start_command=(
                "python",
                "manage.py",
                "runserver",
                "{host}:{port}",
                "--noreload",
            ),
            runtime_http_probe_path="/",
        )
    if name == "fastapi-api":
        entrypoint = _require_framework_entrypoint(metadata, name)
        module_name, _attribute = _parse_module_entrypoint(entrypoint, profile_name=name)
        return ResolvedFrameworkProfile(
            key=name,
            required_artifacts=(_module_to_python_path(module_name),),
            runtime_start_command=(
                "python",
                "-m",
                "uvicorn",
                entrypoint,
                "--host",
                "{host}",
                "--port",
                "{port}",
            ),
            runtime_http_probe_path="/",
        )
    if name == "flask-web":
        entrypoint = _require_framework_entrypoint(metadata, name)
        module_name, attribute = _parse_module_entrypoint(entrypoint, profile_name=name)
        return ResolvedFrameworkProfile(
            key=name,
            required_artifacts=(_module_to_python_path(module_name),),
            runtime_start_command=(
                "python",
                "-c",
                (
                    "import importlib; "
                    f"module = importlib.import_module({module_name!r}); "
                    f"app = getattr(module, {attribute!r}); "
                    "app.run(host='{host}', port={port}, use_reloader=False)"
                ),
            ),
            runtime_http_probe_path="/",
        )
    if name == "node-web":
        return ResolvedFrameworkProfile(
            key=name,
            required_artifacts=("package.json",),
            runtime_http_probe_path="/",
        )
    raise ValueError(f"unsupported metadata.framework_profile: {name}")


def _require_framework_entrypoint(metadata: Mapping[str, Any], profile_name: str) -> str:
    entrypoint = metadata.get("framework_entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError(
            f"metadata.framework_entrypoint is required for framework_profile {profile_name}"
        )
    return entrypoint.strip()


def _parse_module_entrypoint(entrypoint: str, *, profile_name: str) -> tuple[str, str]:
    if ":" not in entrypoint:
        raise ValueError(
            f"metadata.framework_entrypoint for {profile_name} must look like module:attribute"
        )
    module_name, attribute = entrypoint.split(":", 1)
    module_name = module_name.strip()
    attribute = attribute.strip()
    if not module_name or not attribute:
        raise ValueError(
            f"metadata.framework_entrypoint for {profile_name} must look like module:attribute"
        )
    return module_name, attribute


def _module_to_python_path(module_name: str) -> str:
    path = PurePosixPath(*module_name.split(".")).with_suffix(".py")
    return path.as_posix()
