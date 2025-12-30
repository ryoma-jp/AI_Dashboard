"""app URL Configuration for Web API
"""
from django.urls import path
from apis import get_project_list, get_dataset_list, get_job_detail

urlpatterns = [
    path('api/get_project_list', get_project_list.as_view(), name='get_project_list'),
    path('api/get_dataset_list', get_dataset_list.as_view(), name='get_dataset_list'),
    path('api/jobs/<int:pk>/', get_job_detail.as_view(), name='get_job_detail'),
]
