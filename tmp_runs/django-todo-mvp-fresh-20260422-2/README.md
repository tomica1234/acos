# Todoリスト Web アプリ (Django)

シンプルなTodoリストWebアプリです。Djangoで構築されています。

## 要件

- Python 3.8+
- Django 4.x

## セットアップ手順

1. **仮想環境の作成**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Djangoのインストール**

   ```bash
   pip install django
   ```

3. **マイグレーションの実行**

   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

4. **スーパーユーザーの作成（管理画面用）**

   ```bash
   python manage.py createsuperuser
   ```

## 実行手順

```bash
python manage.py runserver
```

ブラウザで `http://127.0.0.1:8000/` にアクセスしてください。

管理画面は `http://127.0.0.1:8000/admin/` からアクセスできます。

## テストの実行

```bash
python manage.py test
```

## 機能

- Todoの一覧表示
- Todoの新規作成
- Todoの完了/未完了切替
- Todoの削除
- Django管理画面からのTodo確認

## 構成

```
acos_todo/
├── manage.py
├── README.md
├── todos/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── admin.py
│   ├── tests.py
│   └── templates/
│       └── todos/
│           └── todo_list.html
└── acos_todo/
    ├── settings.py
    └── urls.py
```
