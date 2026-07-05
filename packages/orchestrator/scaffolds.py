"""Deterministic scaffolds for generation-test jobs."""

from __future__ import annotations

from packages.schemas.agent_outputs import FilePatch, ImplementationResult, TestWriterResult
from packages.schemas.models import ImplementationStatus


def build_scaffold(name: str) -> tuple[ImplementationResult, TestWriterResult] | None:
    if name == "django_todo":
        return build_django_todo_scaffold()
    return None


def build_django_todo_scaffold() -> tuple[ImplementationResult, TestWriterResult]:
    implementation_patches = [
        FilePatch(
            path="manage.py",
            operation="create",
            content="""#!/usr/bin/env python
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "todo_project.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
""",
        ),
        FilePatch(path="todo_project/__init__.py", operation="create", content=""),
        FilePatch(
            path="todo_project/settings.py",
            operation="create",
            content="""from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "dev-secret-key"
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "todos",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "todo_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "todo_project.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "ja"
TIME_ZONE = "Asia/Tokyo"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
""",
        ),
        FilePatch(
            path="todo_project/urls.py",
            operation="create",
            content="""from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("todos.urls")),
]
""",
        ),
        FilePatch(
            path="todo_project/wsgi.py",
            operation="create",
            content="""import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "todo_project.settings")
application = get_wsgi_application()
""",
        ),
        FilePatch(
            path="todo_project/asgi.py",
            operation="create",
            content="""import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "todo_project.settings")
application = get_asgi_application()
""",
        ),
        FilePatch(path="todos/__init__.py", operation="create", content=""),
        FilePatch(
            path="todos/apps.py",
            operation="create",
            content="""from django.apps import AppConfig


class TodosConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "todos"
""",
        ),
        FilePatch(
            path="todos/models.py",
            operation="create",
            content="""from django.db import models


class Todo(models.Model):
    title = models.CharField("タイトル", max_length=200)
    completed = models.BooleanField("完了", default=False)
    created_at = models.DateTimeField("作成日時", auto_now_add=True)

    class Meta:
        ordering = ["completed", "-created_at"]
        verbose_name = "Todo"
        verbose_name_plural = "Todos"

    def __str__(self) -> str:
        return self.title
""",
        ),
        FilePatch(
            path="todos/admin.py",
            operation="create",
            content="""from django.contrib import admin

from .models import Todo


@admin.register(Todo)
class TodoAdmin(admin.ModelAdmin):
    list_display = ("title", "completed", "created_at")
    list_filter = ("completed",)
    search_fields = ("title",)
""",
        ),
        FilePatch(
            path="todos/urls.py",
            operation="create",
            content="""from django.urls import path

from . import views

app_name = "todos"

urlpatterns = [
    path("", views.todo_list, name="list"),
    path("create/", views.todo_create, name="create"),
    path("<int:pk>/toggle/", views.todo_toggle, name="toggle"),
    path("<int:pk>/delete/", views.todo_delete, name="delete"),
]
""",
        ),
        FilePatch(
            path="todos/views.py",
            operation="create",
            content="""from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .models import Todo


def todo_list(request):
    todos = Todo.objects.all()
    return render(request, "todos/index.html", {"todos": todos})


@require_POST
def todo_create(request):
    title = request.POST.get("title", "").strip()
    if title:
        Todo.objects.create(title=title)
    return redirect("todos:list")


@require_POST
def todo_toggle(request, pk):
    todo = get_object_or_404(Todo, pk=pk)
    todo.completed = not todo.completed
    todo.save(update_fields=["completed"])
    return redirect("todos:list")


@require_POST
def todo_delete(request, pk):
    todo = get_object_or_404(Todo, pk=pk)
    todo.delete()
    return redirect("todos:list")
""",
        ),
        FilePatch(
            path="todos/templates/todos/index.html",
            operation="create",
            content="""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Todoリスト</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; }
    form { display: inline; }
    .create { display: flex; gap: 8px; margin-bottom: 24px; }
    .create input { flex: 1; padding: 8px; }
    button { padding: 7px 10px; cursor: pointer; }
    li { display: flex; gap: 8px; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #ddd; }
    .done { color: #777; text-decoration: line-through; }
    .actions { white-space: nowrap; }
  </style>
</head>
<body>
  <h1>Todoリスト</h1>
  <form class="create" method="post" action="{% url 'todos:create' %}">
    {% csrf_token %}
    <input name="title" placeholder="新しいTodoを入力" aria-label="新しいTodo">
    <button type="submit">追加</button>
  </form>

  <ul>
    {% for todo in todos %}
      <li>
        <span class="{% if todo.completed %}done{% endif %}">{{ todo.title }}</span>
        <span class="actions">
          <form method="post" action="{% url 'todos:toggle' todo.pk %}">
            {% csrf_token %}
            <button type="submit">{% if todo.completed %}未完了に戻す{% else %}完了{% endif %}</button>
          </form>
          <form method="post" action="{% url 'todos:delete' todo.pk %}">
            {% csrf_token %}
            <button type="submit">削除</button>
          </form>
        </span>
      </li>
    {% empty %}
      <li>Todoはまだありません。</li>
    {% endfor %}
  </ul>
</body>
</html>
""",
        ),
        FilePatch(path="todos/migrations/__init__.py", operation="create", content=""),
        FilePatch(
            path="todos/migrations/0001_initial.py",
            operation="create",
            content="""from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Todo",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("title", models.CharField(max_length=200, verbose_name="タイトル")),
                ("completed", models.BooleanField(default=False, verbose_name="完了")),
                ("created_at", models.DateTimeField(auto_now_add=True, verbose_name="作成日時")),
            ],
            options={
                "verbose_name": "Todo",
                "verbose_name_plural": "Todos",
                "ordering": ["completed", "-created_at"],
            },
        ),
    ]
""",
        ),
        FilePatch(
            path="README.md",
            operation="create",
            content="""# Django Todoリスト

シンプルな日本語UIのTodoリストWebアプリです。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install Django
python manage.py migrate
```

## 起動

```bash
python manage.py runserver
```

ブラウザで `http://127.0.0.1:8000/` を開きます。

## テスト

```bash
python manage.py test
```

## 管理画面

```bash
python manage.py createsuperuser
python manage.py runserver
```

`http://127.0.0.1:8000/admin/` からTodoを確認できます。
""",
        ),
        FilePatch(path="requirements.txt", operation="create", content="Django\n"),
    ]
    test_patches = [
        FilePatch(
            path="todos/tests.py",
            operation="create",
            content="""from django.test import TestCase
from django.urls import reverse

from .models import Todo


class TodoModelTests(TestCase):
    def test_string_representation_uses_title(self):
        todo = Todo.objects.create(title="買い物")
        self.assertEqual(str(todo), "買い物")


class TodoViewTests(TestCase):
    def test_list_page_shows_empty_message(self):
        response = self.client.get(reverse("todos:list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Todoはまだありません。")

    def test_create_todo(self):
        response = self.client.post(reverse("todos:create"), {"title": "牛乳を買う"})
        self.assertRedirects(response, reverse("todos:list"))
        self.assertTrue(Todo.objects.filter(title="牛乳を買う").exists())

    def test_toggle_todo(self):
        todo = Todo.objects.create(title="掃除")
        response = self.client.post(reverse("todos:toggle", args=[todo.pk]))
        self.assertRedirects(response, reverse("todos:list"))
        todo.refresh_from_db()
        self.assertTrue(todo.completed)

    def test_delete_todo(self):
        todo = Todo.objects.create(title="削除する")
        response = self.client.post(reverse("todos:delete", args=[todo.pk]))
        self.assertRedirects(response, reverse("todos:list"))
        self.assertFalse(Todo.objects.filter(pk=todo.pk).exists())
""",
        )
    ]
    return (
        ImplementationResult(
            status=ImplementationStatus.IMPLEMENTED,
            summary="Created a minimal Django Todo application scaffold.",
            changed_files=[patch.path for patch in implementation_patches],
            patches=implementation_patches,
        ),
        TestWriterResult(
            summary="Added Django tests for model and Todo CRUD views.",
            changed_files=[patch.path for patch in test_patches],
            patches=test_patches,
            test_strategy=["Run python manage.py test"],
        ),
    )
