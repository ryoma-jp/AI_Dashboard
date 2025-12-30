"""Serializers
"""
from rest_framework import serializers
from app.models import Project, Dataset, OperationJob, OperationStep

class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ['name', 'id']

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['name', 'project', 'dataset_dir_offset']


class OperationStepSerializer(serializers.ModelSerializer):
    class Meta:
        model = OperationStep
        fields = ['order', 'label', 'status', 'detail']


class OperationJobSerializer(serializers.ModelSerializer):
    steps = OperationStepSerializer(many=True, read_only=True)

    class Meta:
        model = OperationJob
        fields = [
            'id',
            'job_type',
            'status',
            'message',
            'project',
            'model',
            'dataset',
            'created_at',
            'updated_at',
            'steps',
        ]
