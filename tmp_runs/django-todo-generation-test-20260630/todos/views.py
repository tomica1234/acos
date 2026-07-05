from django.shortcuts import get_object_or_404, redirect, render
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
