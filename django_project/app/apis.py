"""Web APIs
"""
from app.models import Project, Dataset
from rest_framework.generics import ListCreateAPIView
from serializers import ProjectSerializer, DatasetSerializer

class get_project_list(ListCreateAPIView):
    # 対象とするモデルのオブジェクトを定義
    queryset = Project.objects.all()
    
    # APIがデータを返すためのデータ変換ロジックを定義
    serializer_class = ProjectSerializer
    
    # 認証
    permission_classes = []
    
    # HTTPリクエストメソッドを制限
    http_method_names = ['get']

class get_dataset_list(ListCreateAPIView):
    # 対象とするモデルのオブジェクトを定義
    queryset = Dataset.objects.filter(download_status='Done')
    
    # APIがデータを返すためのデータ変換ロジックを定義
    serializer_class = DatasetSerializer
    
    # 認証
    permission_classes = []
    
    # HTTPリクエストメソッドを制限
    http_method_names = ['get']

