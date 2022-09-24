"""app URL Configuration for Web API
"""
from django.urls import path
from apis import api

urlpatterns = [
    path('api', api.as_view(), name='api'),
]
