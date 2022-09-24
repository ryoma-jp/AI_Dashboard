"""Serializers
"""
from rest_framework import serializers
from app.models import Project                              # モデル呼出

class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ['name']

