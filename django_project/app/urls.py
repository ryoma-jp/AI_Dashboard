"""app URL Configuration
"""
from django.urls import path
from . import views_index, views_project, views_dataset, views_training, views_inference

urlpatterns = [
    path('', views_index.index, name='index'),
    path('dataset/', views_dataset.dataset, name='dataset'),
    path('dataset_detail/p<int:project_id>d<int:dataset_id>/', views_dataset.dataset_detail, name='dataset_detail'),
    path('training/', views_training.training, name='training'),
    path('training/model_paraemter_edit/<int:model_id>', views_training.model_paraemter_edit, name='model_paraemter_edit'),
    path('project_new/', views_project.project_new, name='project_new'),
    path('project<int:project_id>_edit/', views_project.project_edit, name='project_edit'),
    path('project<int:project_id>/model_new/', views_project.model_new, name='model_new'),
    path('project<int:project_id>/model<int:model_id>_edit/', views_project.model_edit, name='model_edit'),
    path('inference/', views_inference.inference, name='inference'),
]
