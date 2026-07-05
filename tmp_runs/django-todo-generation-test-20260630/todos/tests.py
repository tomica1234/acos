from django.test import TestCase
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
