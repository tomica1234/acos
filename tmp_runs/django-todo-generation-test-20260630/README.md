# Django Todoリスト

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
