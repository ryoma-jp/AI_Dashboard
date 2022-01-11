from django.db import models
from django.conf import settings

# Create your models here.

#---------------------------------------
# クラス：プロジェクト
#---------------------------------------
class Project(models.Model):
    name = models.CharField('ProjectName', max_length=128)
    description = models.TextField('Description', blank=True)
    
    # --- for Dataset view ---
    dataset_view_selected = models.CharField('Selected Status in Dataset View',
                                max_length=10,
                                default='unchecked')  # checked or unchecked
    
    def __str__(self):
        return self.name

#---------------------------------------
# クラス：データセット
#---------------------------------------
class Dataset(models.Model):
    name = models.CharField('DatasetName', max_length=128)
    project = models.ForeignKey(Project, verbose_name='Project', on_delete=models.CASCADE)
    
    train_zip = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    train_csv = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    valid_zip = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    valid_csv = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    test_zip = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    test_csv = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

#---------------------------------------
# クラス：MlModel
#---------------------------------------
class MlModel(models.Model):
    name = models.CharField('ModelName', max_length=128)
    description = models.TextField('Description', blank=True)
    project = models.ForeignKey(Project, verbose_name='Project', on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, verbose_name='Dataset', on_delete=models.CASCADE)
    
    model_dir = models.TextField('ModelDir', blank=True)
    
    STAT_IDLE = 'IDLE'
    STAT_TRAINING = 'TRAINING'
    STAT_DONE = 'DONE'
    status = models.TextField('Status')
    
    def __str__(self):
        return self.name
