from django.db import models

# Create your models here.

#---------------------------------------
# クラス：データセットファイル
#---------------------------------------
class DatasetFile(models.Model):
    train_zip = models.FileField(upload_to='dataset/')
    train_csv = models.FileField(upload_to='dataset/')
    valid_zip = models.FileField(upload_to='dataset/')
    valid_csv = models.FileField(upload_to='dataset/')
    test_zip = models.FileField(upload_to='dataset/')
    test_csv = models.FileField(upload_to='dataset/')
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

