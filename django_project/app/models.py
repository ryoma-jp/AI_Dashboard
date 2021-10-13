from django.db import models

# Create your models here.

#---------------------------------------
# クラス：データセットファイル
#---------------------------------------
class DatasetFile(models.Model):
    x_train = models.FileField(upload_to='dataset/')
    y_train = models.FileField(upload_to='dataset/')
    x_valid = models.FileField(upload_to='dataset/')
    y_valid = models.FileField(upload_to='dataset/')
    x_test = models.FileField(upload_to='dataset/')
    y_test = models.FileField(upload_to='dataset/')
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

