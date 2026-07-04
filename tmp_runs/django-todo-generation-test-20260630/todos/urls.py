from django.urls import path

from . import views

app_name = "todos"

urlpatterns = [
    path("", views.todo_list, name="list"),
    path("create/", views.todo_create, name="create"),
    path("<int:pk>/toggle/", views.todo_toggle, name="toggle"),
    path("<int:pk>/delete/", views.todo_delete, name="delete"),
]
