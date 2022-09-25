"""Serializers
"""
from rest_framework import serializers
from app.models import Project, Dataset

class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ['name', 'id']

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['name', 'project']
