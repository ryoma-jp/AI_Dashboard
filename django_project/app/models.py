from django.db import models
from django.conf import settings

# Create your models here.

#---------------------------------------
# クラス：データセットファイル
#---------------------------------------
class DatasetFile(models.Model):
    train_zip = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    train_csv = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    valid_zip = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    valid_csv = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    test_zip = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    test_csv = models.FileField(upload_to=getattr(settings, 'DATASET_DIR', None))
    uploaded_at = models.DateTimeField(auto_now_add=True)

#---------------------------------------
# クラス：データセット選択用
#---------------------------------------
class DatasetSelection(models.Model):
    DATASET_CHOICES = (
        ("MNIST", "MNIST"),
        ("CIFAR-10", "CIFAR-10"),
        ("User data", "User data"),
    )
    
    selection = models.CharField(verbose_name="Dataset Selection", choices=DATASET_CHOICES, default=DATASET_CHOICES[0][0], max_length=50)

