"""app URL Configuration
"""
from django.urls import path, include
from . import views_index, views_project, views_dataset, views_training, views_inference, views_view_streaming, views_ai_model_sdk

urlpatterns = [
    path('', views_index.index, name='index'),
    path('dataset/', views_dataset.dataset, name='dataset'),
    path('dataset_detail/p<int:project_id>d<int:dataset_id>/', views_dataset.dataset_detail, name='dataset_detail'),
    path('ai_model_sdk/', views_ai_model_sdk.ai_model_sdk, name='ai_model_sdk'),
    path('training/', views_training.training, name='training'),
    path('training/model_parameter_edit/<int:model_id>', views_training.model_parameter_edit, name='model_parameter_edit'),
    path('project_new/', views_project.project_new, name='project_new'),
    path('project<int:project_id>_edit/', views_project.project_edit, name='project_edit'),
    path('project<int:project_id>/model_new/', views_project.model_new, name='model_new'),
    path('project<int:project_id>/model<int:model_id>_edit/', views_project.model_edit, name='model_edit'),
    path('inference/', views_inference.inference, name='inference'),
    path('inference/download', views_inference.download_prediction, name='download_prediction'),
    path('view_streaming/', views_view_streaming.view_streaming, name='view_streaming'),
    path('view_streaming/usb_cam', views_view_streaming.usb_cam, name='view_streaming_usb_cam'),
    path('view_streaming/youtube', views_view_streaming.youtube, name='view_streaming_youtube'),
    path('', include('urls_api')),
]
