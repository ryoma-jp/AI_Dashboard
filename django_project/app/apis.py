"""Web APIs
"""
from app.models import Project                          # モデル呼出
from rest_framework.generics import ListCreateAPIView   # API
from serializers import ProjectSerializer               # APIで渡すデータをJSON,XML変換

class api(ListCreateAPIView):
    # 対象とするモデルのオブジェクトを定義
    queryset = Project.objects.all()
    
    # APIがデータを返すためのデータ変換ロジックを定義
    serializer_class = ProjectSerializer
    
    # 認証
    permission_classes = []
    
    # HTTPリクエストメソッドを制限
    http_method_names = ['get']

