from django.db import migrations, models


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
