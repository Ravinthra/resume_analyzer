"""URL Configuration for the Analyzer app."""
from django.urls import path
from . import views

app_name = "analyzer"

urlpatterns = [
    path("", views.home, name="home"),
    path("analyze/", views.analyze, name="analyze"),
]
