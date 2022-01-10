"""app URL Configuration
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dataset/', views.dataset, name='dataset'),
    path('training/', views.training, name='training'),
    path('project_new/', views.project_new, name='project_new'),
    path('project<int:project_id>/model_new/', views.model_new, name='model_new'),
]
