"""Deterministic framework bootstrap scaffolds for ACOS jobs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from packages.orchestrator.framework_profiles import resolve_framework_profile
from packages.schemas.agent_outputs import FilePatch

_PYTHON_IDENTIFIER_PATTERN = re.compile(r"[^A-Za-z0-9_]+")
_PACKAGE_NAME_PATTERN = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class ResolvedFrameworkScaffold:
    key: str
    required_artifacts: tuple[str, ...]
    patches: tuple[FilePatch, ...]


def resolve_framework_scaffold(
    metadata: Mapping[str, Any],
    *,
    workspace_root: str | Path,
) -> ResolvedFrameworkScaffold | None:
    profile = resolve_framework_profile(metadata)
    if profile is None:
        return None
    workspace_name = Path(workspace_root).resolve().name
    if profile.key == "django-web":
        return _resolve_django_scaffold(metadata, workspace_name=workspace_name)
    if profile.key == "fastapi-api":
        return _resolve_fastapi_scaffold(metadata)
    if profile.key == "flask-web":
        return _resolve_flask_scaffold(metadata)
    if profile.key == "node-web":
        return _resolve_node_scaffold(workspace_name=workspace_name)
    raise ValueError(f"unsupported scaffold for framework_profile {profile.key}")


def _resolve_django_scaffold(
    metadata: Mapping[str, Any],
    *,
    workspace_name: str,
) -> ResolvedFrameworkScaffold:
    project_name = _python_identifier(
        str(metadata.get("framework_project_name") or workspace_name)
    )
    required_artifacts = (
        "manage.py",
        f"{project_name}/__init__.py",
        f"{project_name}/settings.py",
        f"{project_name}/urls.py",
        f"{project_name}/wsgi.py",
    )
    patches = (
        FilePatch(
            path="manage.py",
            operation="create",
            rationale="Bootstrap Django management entrypoint.",
            content=(
                "#!/usr/bin/env python\n"
                "import os\n"
                "import sys\n"
                "\n"
                "\n"
                "def main() -> None:\n"
                f'    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "{project_name}.settings")\n'
                "    from django.core.management import execute_from_command_line\n"
                "\n"
                "    execute_from_command_line(sys.argv)\n"
                "\n"
                "\n"
                'if __name__ == "__main__":\n'
                "    main()\n"
            ),
        ),
        FilePatch(
            path=f"{project_name}/__init__.py",
            operation="create",
            rationale="Mark the Django project package as importable.",
            content="",
        ),
        FilePatch(
            path=f"{project_name}/settings.py",
            operation="create",
            rationale="Provide minimal Django settings for local development.",
            content=(
                '"""Django settings for the generated ACOS project."""\n'
                "\n"
                "from pathlib import Path\n"
                "\n"
                "BASE_DIR = Path(__file__).resolve().parent.parent\n"
                '\n'
                'SECRET_KEY = "acos-dev-secret-key"\n'
                "\n"
                "DEBUG = True\n"
                "\n"
                'ALLOWED_HOSTS = ["*"]\n'
                "\n"
                "INSTALLED_APPS = [\n"
                '    "django.contrib.admin",\n'
                '    "django.contrib.auth",\n'
                '    "django.contrib.contenttypes",\n'
                '    "django.contrib.sessions",\n'
                '    "django.contrib.messages",\n'
                '    "django.contrib.staticfiles",\n'
                "]\n"
                "\n"
                "MIDDLEWARE = [\n"
                '    "django.middleware.security.SecurityMiddleware",\n'
                '    "django.contrib.sessions.middleware.SessionMiddleware",\n'
                '    "django.middleware.common.CommonMiddleware",\n'
                '    "django.middleware.csrf.CsrfViewMiddleware",\n'
                '    "django.contrib.auth.middleware.AuthenticationMiddleware",\n'
                '    "django.contrib.messages.middleware.MessageMiddleware",\n'
                '    "django.middleware.clickjacking.XFrameOptionsMiddleware",\n'
                "]\n"
                "\n"
                f'ROOT_URLCONF = "{project_name}.urls"\n'
                "\n"
                "TEMPLATES = [\n"
                "    {\n"
                '        "BACKEND": "django.template.backends.django.DjangoTemplates",\n'
                '        "DIRS": [BASE_DIR / "templates"],\n'
                '        "APP_DIRS": True,\n'
                '        "OPTIONS": {\n'
                '            "context_processors": [\n'
                '                "django.template.context_processors.request",\n'
                '                "django.contrib.auth.context_processors.auth",\n'
                '                "django.contrib.messages.context_processors.messages",\n'
                "            ],\n"
                "        },\n"
                "    },\n"
                "]\n"
                "\n"
                f'WSGI_APPLICATION = "{project_name}.wsgi.application"\n'
                "\n"
                "DATABASES = {\n"
                '    "default": {\n'
                '        "ENGINE": "django.db.backends.sqlite3",\n'
                '        "NAME": BASE_DIR / "db.sqlite3",\n'
                "    }\n"
                "}\n"
                "\n"
                'LANGUAGE_CODE = "en-us"\n'
                'TIME_ZONE = "UTC"\n'
                "USE_I18N = True\n"
                "USE_TZ = True\n"
                "\n"
                'STATIC_URL = "static/"\n'
                '\n'
                'DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"\n'
            ),
        ),
        FilePatch(
            path=f"{project_name}/urls.py",
            operation="create",
            rationale="Provide a root URL configuration so Django can start.",
            content=(
                "from django.contrib import admin\n"
                "from django.urls import path\n"
                "\n"
                "urlpatterns = [\n"
                '    path("admin/", admin.site.urls),\n'
                "]\n"
            ),
        ),
        FilePatch(
            path=f"{project_name}/wsgi.py",
            operation="create",
            rationale="Provide the default Django WSGI entrypoint.",
            content=(
                "import os\n"
                "\n"
                "from django.core.wsgi import get_wsgi_application\n"
                "\n"
                f'os.environ.setdefault("DJANGO_SETTINGS_MODULE", "{project_name}.settings")\n'
                "\n"
                "application = get_wsgi_application()\n"
            ),
        ),
    )
    return ResolvedFrameworkScaffold(
        key="django-web",
        required_artifacts=required_artifacts,
        patches=patches,
    )


def _resolve_fastapi_scaffold(metadata: Mapping[str, Any]) -> ResolvedFrameworkScaffold:
    module_name, attribute = _parse_module_entrypoint(metadata, profile_name="fastapi-api")
    module_path = _module_to_python_path(module_name)
    init_artifacts = tuple(_package_init_artifacts(module_name))
    required_artifacts = (*init_artifacts, module_path)
    patches = [
        *[
            FilePatch(
                path=path,
                operation="create",
                rationale="Mark the Python package as importable.",
                content="",
            )
            for path in init_artifacts
        ],
        FilePatch(
            path=module_path,
            operation="create",
            rationale="Bootstrap a minimal FastAPI application entrypoint.",
            content=(
                "from fastapi import FastAPI\n"
                "\n"
                f"{attribute} = FastAPI()\n"
                "\n"
                f"@{attribute}.get('/')\n"
                "def read_root() -> dict[str, str]:\n"
                "    return {'status': 'ok'}\n"
            ),
        ),
    ]
    return ResolvedFrameworkScaffold(
        key="fastapi-api",
        required_artifacts=required_artifacts,
        patches=tuple(patches),
    )


def _resolve_flask_scaffold(metadata: Mapping[str, Any]) -> ResolvedFrameworkScaffold:
    module_name, attribute = _parse_module_entrypoint(metadata, profile_name="flask-web")
    module_path = _module_to_python_path(module_name)
    init_artifacts = tuple(_package_init_artifacts(module_name))
    required_artifacts = (*init_artifacts, module_path)
    patches = [
        *[
            FilePatch(
                path=path,
                operation="create",
                rationale="Mark the Python package as importable.",
                content="",
            )
            for path in init_artifacts
        ],
        FilePatch(
            path=module_path,
            operation="create",
            rationale="Bootstrap a minimal Flask application entrypoint.",
            content=(
                "from flask import Flask\n"
                "\n"
                f"{attribute} = Flask(__name__)\n"
                "\n"
                f"@{attribute}.get('/')\n"
                "def read_root() -> dict[str, str]:\n"
                "    return {'status': 'ok'}\n"
            ),
        ),
    ]
    return ResolvedFrameworkScaffold(
        key="flask-web",
        required_artifacts=required_artifacts,
        patches=tuple(patches),
    )


def _resolve_node_scaffold(*, workspace_name: str) -> ResolvedFrameworkScaffold:
    package_name = _package_name(workspace_name)
    return ResolvedFrameworkScaffold(
        key="node-web",
        required_artifacts=("package.json",),
        patches=(
            FilePatch(
                path="package.json",
                operation="create",
                rationale="Bootstrap a minimal Node package manifest.",
                content=json.dumps(
                    {
                        "name": package_name,
                        "private": True,
                        "version": "0.1.0",
                        "scripts": {
                            "start": "node index.js",
                            "test": "node --test",
                        },
                    },
                    indent=2,
                )
                + "\n",
            ),
        ),
    )


def _parse_module_entrypoint(
    metadata: Mapping[str, Any],
    *,
    profile_name: str,
) -> tuple[str, str]:
    raw = metadata.get("framework_entrypoint")
    if not isinstance(raw, str) or ":" not in raw:
        raise ValueError(
            f"metadata.framework_entrypoint for {profile_name} must look like module:attribute"
        )
    module_name, attribute = raw.split(":", 1)
    module_name = module_name.strip()
    attribute = attribute.strip()
    if not module_name or not attribute:
        raise ValueError(
            f"metadata.framework_entrypoint for {profile_name} must look like module:attribute"
        )
    return module_name, attribute


def _module_to_python_path(module_name: str) -> str:
    return PurePosixPath(*module_name.split(".")).with_suffix(".py").as_posix()


def _package_init_artifacts(module_name: str) -> list[str]:
    parts = PurePosixPath(*module_name.split(".")).parts[:-1]
    artifacts: list[str] = []
    prefix: list[str] = []
    for part in parts:
        prefix.append(part)
        artifacts.append(PurePosixPath(*prefix, "__init__.py").as_posix())
    return artifacts


def _python_identifier(raw: str) -> str:
    normalized = _PYTHON_IDENTIFIER_PATTERN.sub("_", raw).strip("_")
    if not normalized:
        return "acos_project"
    if normalized[0].isdigit():
        normalized = f"app_{normalized}"
    return normalized


def _package_name(raw: str) -> str:
    normalized = _PACKAGE_NAME_PATTERN.sub("-", raw.strip().lower()).strip("-")
    return normalized or "acos-app"
