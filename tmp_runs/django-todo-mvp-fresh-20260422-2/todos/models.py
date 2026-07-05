from django.db import models


class Todo(models.Model):
    title = models.CharField(max_length=200)
    completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        status = "完了" if self.completed else "未完了"
        return f"{self.title} ({status})"

    class Meta:
        ordering = ['-created_at']
