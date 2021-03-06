import os

from django.db import models
from django.conf import settings

# Create your models here.

#---------------------------------------
# クラス：プロジェクト
#---------------------------------------
class Project(models.Model):
    name = models.CharField('ProjectName', max_length=128)
    description = models.TextField('Description', blank=True)
    hash = models.CharField('Project hash', max_length=128)
    
    def __str__(self):
        return self.name

#---------------------------------------
# クラス：データセット
#---------------------------------------
def train_dataset_path(instance, filename):
    base_dir = getattr(settings, 'DATASET_DIR', None)
    return os.path.join(base_dir, f'{instance.project.hash}', f'dataset_{instance.id}', 'train', filename)
    
def validation_dataset_path(instance, filename):
    base_dir = getattr(settings, 'DATASET_DIR', None)
    return os.path.join(base_dir, f'{instance.project.hash}', f'dataset_{instance.id}', 'validation', filename)
    
def test_dataset_path(instance, filename):
    base_dir = getattr(settings, 'DATASET_DIR', None)
    return os.path.join(base_dir, f'{instance.project.hash}', f'dataset_{instance.id}', 'test', filename)
    
class Dataset(models.Model):
    name = models.CharField('DatasetName', max_length=128)
    project = models.ForeignKey(Project, verbose_name='Project', on_delete=models.CASCADE)
    
    dataset_dir = models.CharField('Dataset directory in the Project directory', max_length=512, blank=True)
    
    train_zip = models.FileField(upload_to=train_dataset_path, max_length=512)
    valid_zip = models.FileField(upload_to=validation_dataset_path, max_length=512)
    test_zip = models.FileField(upload_to=test_dataset_path, max_length=512)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    STATUS_NONE = 'None'
    STATUS_PREPARING = 'Preparing'
    STATUS_PROCESSING = 'Processing'
    STATUS_DONE = 'Done'
    STATUS = (
        (STATUS_NONE, STATUS_NONE),
        (STATUS_PREPARING, STATUS_PREPARING),
        (STATUS_PROCESSING, STATUS_PROCESSING),
        (STATUS_DONE, STATUS_DONE),
    )
    download_status = models.CharField(max_length=32, choices=STATUS, default=STATUS_NONE)
    image_gallery_status = models.CharField(max_length=32, choices=STATUS, default=STATUS_NONE)
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if (self.id is None):
            # --- dummy save, set id and zip files ---
            _tmp_train_zip = self.train_zip
            _tmp_valid_zip = self.valid_zip
            _tmp_test_zip = self.test_zip
            
            self.train_zip = None
            self.valid_zip = None
            self.test_zip = None
            super().save(*args, **kwargs)
            
            self.train_zip = _tmp_train_zip
            self.valid_zip = _tmp_valid_zip
            self.test_zip = _tmp_test_zip
            if ('force_insert' in kwargs):
                kwargs.pop('force_insert')
        
            # --- set dataset_dir ---
            self.dataset_dir = os.path.join(
                                   getattr(settings, 'MEDIA_ROOT', None),
                                   getattr(settings, 'DATASET_DIR', None),
                                   self.project.hash,
                                   f'dataset_{self.id}')
            
        super().save(*args, **kwargs)

#---------------------------------------
# クラス：MlModel
#---------------------------------------
class MlModel(models.Model):
    name = models.CharField('ModelName', max_length=128)
    description = models.TextField('Description', blank=True)
    hash = models.CharField('Model hash', max_length=128)
    project = models.ForeignKey(Project, verbose_name='Project', on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, verbose_name='Dataset', on_delete=models.SET_NULL, null=True)
    dataset_pickle = models.CharField('Dataset Object (*.pkl)', max_length=512, blank=True)
    
    model_dir = models.CharField('Model Directory', max_length=1024)
    
    STAT_IDLE = 'IDLE'
    STAT_TRAINING = 'TRAINING'
    STAT_DONE = 'DONE'
    status = models.TextField('Status')
    
    # --- for Training control ---
    training_pid = models.IntegerField('Training worker PID', null=True, default=None)
    tensorboard_pid = models.IntegerField('Tensorboard worker PID', null=True, default=None)
    
    def __str__(self):
        return self.name
