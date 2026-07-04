from django.test import TestCase, Client
from django.urls import reverse
from .models import Todo


class TodoListViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.todo = Todo.objects.create(title="テストTodo", completed=False)

    def test_get_todo_list(self):
        response = self.client.get(reverse('todo_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "テストTodo")

    def test_get_todo_list_empty(self):
        # 空のリストでもエラーにならないことを確認
        Todo.objects.all().delete()
        response = self.client.get(reverse('todo_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Todoがありません")

    def test_get_todo_list_multiple(self):
        # 複数のTodoが一覧に表示されることを確認
        Todo.objects.create(title="Todo 2", completed=True)
        Todo.objects.create(title="Todo 3", completed=False)
        response = self.client.get(reverse('todo_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "テストTodo")
        self.assertContains(response, "Todo 2")
        self.assertContains(response, "Todo 3")


class TodoCreateViewTest(TestCase):
    def test_create_todo(self):
        response = self.client.post(reverse('todo_create'), {
            'title': '新しいTodo'
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Todo.objects.count(), 1)
        self.assertEqual(Todo.objects.first().title, '新しいTodo')

    def test_create_todo_empty_title(self):
        # 空のタイトルは作成されないことを確認
        response = self.client.post(reverse('todo_create'), {
            'title': ''
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Todo.objects.count(), 0)

    def test_create_todo_whitespace_only(self):
        # 空白のみのタイトルは作成されないことを確認
        response = self.client.post(reverse('todo_create'), {
            'title': '   '
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Todo.objects.count(), 0)

    def test_create_todo_with_spaces(self):
        # 前後の空白がトリムされることを確認
        response = self.client.post(reverse('todo_create'), {
            'title': '  タイトル  '
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Todo.objects.count(), 1)
        self.assertEqual(Todo.objects.first().title, 'タイトル')


class TodoToggleViewTest(TestCase):
    def setUp(self):
        self.todo = Todo.objects.create(title="テストTodo", completed=False)

    def test_toggle_todo(self):
        response = self.client.post(reverse('todo_toggle', args=[self.todo.pk]))
        self.assertEqual(response.status_code, 302)
        self.todo.refresh_from_db()
        self.assertTrue(self.todo.completed)

    def test_toggle_todo_back_to_uncompleted(self):
        # 完了から未完了に戻せることを確認
        self.todo.completed = True
        self.todo.save()
        response = self.client.post(reverse('todo_toggle', args=[self.todo.pk]))
        self.assertEqual(response.status_code, 302)
        self.todo.refresh_from_db()
        self.assertFalse(self.todo.completed)

    def test_toggle_todo_nonexistent(self):
        # 存在しないIDを指定すると404になることを確認
        response = self.client.post(reverse('todo_toggle', args=[99999]))
        self.assertEqual(response.status_code, 404)


class TodoDeleteViewTest(TestCase):
    def setUp(self):
        self.todo = Todo.objects.create(title="テストTodo", completed=False)

    def test_delete_todo(self):
        response = self.client.post(reverse('todo_delete', args=[self.todo.pk]))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Todo.objects.count(), 0)

    def test_delete_todo_nonexistent(self):
        # 存在しないIDを指定すると404になることを確認
        response = self.client.post(reverse('todo_delete', args=[99999]))
        self.assertEqual(response.status_code, 404)


class TodoModelTest(TestCase):
    def test_todo_creation(self):
        todo = Todo.objects.create(title="モデルテスト", completed=False)
        self.assertEqual(todo.title, "モデルテスト")
        self.assertFalse(todo.completed)
        self.assertIsNotNone(todo.created_at)

    def test_todo_completed_default(self):
        todo = Todo.objects.create(title="デフォルトテスト")
        self.assertFalse(todo.completed)

    def test_todo_str_completed(self):
        todo = Todo.objects.create(title="完了テスト", completed=True)
        self.assertEqual(str(todo), "完了テスト (完了)")

    def test_todo_str_uncompleted(self):
        todo = Todo.objects.create(title="未完了テスト", completed=False)
        self.assertEqual(str(todo), "未完了テスト (未完了)")

    def test_todo_ordering(self):
        todo1 = Todo.objects.create(title="古いTodo", completed=False)
        import time
        time.sleep(0.1)  # created_atの順序を明確にする
        todo2 = Todo.objects.create(title="新しいTodo", completed=False)
        todos = Todo.objects.all()
        self.assertEqual(todos[0], todo2)
        self.assertEqual(todos[1], todo1)
