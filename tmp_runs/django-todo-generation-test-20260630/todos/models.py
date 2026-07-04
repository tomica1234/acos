from django.db import models


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
