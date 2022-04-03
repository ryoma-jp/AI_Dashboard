"""app URL Configuration
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dataset/', views.dataset, name='dataset'),
    path('dataset_detail/p<int:project_id>d<int:dataset_id>/', views.dataset_detail, name='dataset_detail'),
    path('training/', views.training, name='training'),
    path('training/model_paraemter_edit/<int:model_id>', views.model_paraemter_edit, name='model_paraemter_edit'),
    path('project_new/', views.project_new, name='project_new'),
    path('project<int:project_id>_edit/', views.project_edit, name='project_edit'),
    path('project<int:project_id>/model_new/', views.model_new, name='model_new'),
    path('project<int:project_id>/model<int:model_id>_edit/', views.model_edit, name='model_edit'),
]
