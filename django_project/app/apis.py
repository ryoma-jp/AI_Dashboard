"""Web APIs
"""
from app.models import Project, Dataset, OperationJob
from rest_framework.generics import ListCreateAPIView, RetrieveAPIView
from serializers import ProjectSerializer, DatasetSerializer, OperationJobSerializer
from views_common import get_all_fifo_command

class get_project_list(ListCreateAPIView):
    """get_project_list
    
    プロジェクトリストを取得する
    """
    # 対象とするモデルのオブジェクトを定義
    queryset = Project.objects.all()
    
    # APIがデータを返すためのデータ変換ロジックを定義
    serializer_class = ProjectSerializer
    
    # 認証
    permission_classes = []
    
    # HTTPリクエストメソッドを制限
    http_method_names = ['get']

class get_dataset_list(ListCreateAPIView):
    """get_project_list
    
    データセットのリストを取得する
    """
    # 対象とするモデルのオブジェクトを定義
    queryset = Dataset.objects.filter(download_status='Done')
    
    # APIがデータを返すためのデータ変換ロジックを定義
    serializer_class = DatasetSerializer
    
    # 認証
    permission_classes = []
    
    # HTTPリクエストメソッドを制限
    http_method_names = ['get']


class get_job_detail(RetrieveAPIView):
    """get_job_detail
    
    ジョブの進捗（ステップ一覧）を取得する
    """
    queryset = OperationJob.objects.prefetch_related('steps').all()
    serializer_class = OperationJobSerializer
    permission_classes = []
    http_method_names = ['get']

    def get(self, request, *args, **kwargs):
        # FIFOの完了通知を定期ポーリングで拾い、ジョブ状態を更新する
        get_all_fifo_command()
        return super().get(request, *args, **kwargs)

