from django.db import models
from django.conf import settings

# Create your models here.

#---------------------------------------
# クラス：プロジェクト
#---------------------------------------
class Project(models.Model):
    name = models.CharField('ProjectName', max_length=128)
    description = models.TextField('Description', blank=True)
    
    registerd_dataset = [
        ("MNIST", "MNIST"),
        ("CIFAR-10", "CIFAR-10"),
    ]
    
    def __str__(self):
        return self.name

#---------------------------------------
# クラス：データセット
#---------------------------------------
class CustomDataset(models.Model):
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
    
    model_dir = models.TextField('ModelDir', blank=True)
    
    MODEL_STATUS = (
        ("IDLE", "IDLE"),
        ("TRAINING", "TRAINING"),
        ("DONE", "DONE"),
    )
    status = models.CharField(verbose_name="Model Status", choices=MODEL_STATUS, default=MODEL_STATUS[0][0], max_length=64)
    
    def __str__(self):
        return self.name
