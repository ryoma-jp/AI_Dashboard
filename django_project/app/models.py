from pathlib import Path
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
    return Path(base_dir, f'{instance.project.hash}', f'dataset_{instance.id}', 'train', filename)
    
def validation_dataset_path(instance, filename):
    base_dir = getattr(settings, 'DATASET_DIR', None)
    return Path(base_dir, f'{instance.project.hash}', f'dataset_{instance.id}', 'validation', filename)
    
def test_dataset_path(instance, filename):
    base_dir = getattr(settings, 'DATASET_DIR', None)
    return Path(base_dir, f'{instance.project.hash}', f'dataset_{instance.id}', 'test', filename)
    
def meta_dataset_path(instance, filename):
    base_dir = getattr(settings, 'DATASET_DIR', None)
    return Path(base_dir, f'{instance.project.hash}', f'dataset_{instance.id}', 'meta', filename)
    
class Dataset(models.Model):
    name = models.CharField('DatasetName', max_length=128)
    project = models.ForeignKey(Project, verbose_name='Project', on_delete=models.CASCADE)
    
    DATASET_TYPE_NONE = 'None'
    DATASET_TYPE_IMAGE = 'Image'
    DATASET_TYPE_TABLE = 'Table'
    DATASET_TYPE_TIMESERIES = 'Time Series'
    DATASET_TYPE = (
        (DATASET_TYPE_NONE, DATASET_TYPE_NONE),
        (DATASET_TYPE_IMAGE, DATASET_TYPE_IMAGE),
        (DATASET_TYPE_TABLE, DATASET_TYPE_TABLE),
        (DATASET_TYPE_TIMESERIES, DATASET_TYPE_TIMESERIES),
    )
    dataset_type = models.CharField(max_length=32, choices=DATASET_TYPE, default=DATASET_TYPE_NONE)
    
    dataset_dir = models.CharField('Dataset directory in the Project directory', max_length=512, blank=True)
    dataset_dir_offset = models.CharField('', max_length=512, blank=True)    # dataset directory under 'DATASET_DIR'
    
    train_zip = models.FileField(upload_to=train_dataset_path, max_length=512)
    valid_zip = models.FileField(upload_to=validation_dataset_path, max_length=512, blank=True, null=True)
    test_zip = models.FileField(upload_to=test_dataset_path, max_length=512, blank=True, null=True)
    meta_zip = models.FileField(upload_to=meta_dataset_path, max_length=512, blank=True, null=True)
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
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if (self.id is None):
            # --- dummy save, set id and zip files ---
            _tmp_train_zip = self.train_zip
            _tmp_valid_zip = self.valid_zip
            _tmp_test_zip = self.test_zip
            _tmp_meta_zip = self.meta_zip
            
            self.train_zip = None
            self.valid_zip = None
            self.test_zip = None
            self.meta_zip = None
            super().save(*args, **kwargs)
            
            self.train_zip = _tmp_train_zip
            self.valid_zip = _tmp_valid_zip
            self.test_zip = _tmp_test_zip
            self.meta_zip = _tmp_meta_zip
            if ('force_insert' in kwargs):
                kwargs.pop('force_insert')
        
            # --- set dataset_dir ---
            self.dataset_dir = Path(
                                   getattr(settings, 'MEDIA_ROOT', None),
                                   getattr(settings, 'DATASET_DIR', None),
                                   self.project.hash,
                                   f'dataset_{self.id}')
            self.dataset_dir_offset = Path(
                                   self.project.hash,
                                   f'dataset_{self.id}')
            
        super().save(*args, **kwargs)

#---------------------------------------
# クラス：AI Model SDK
#---------------------------------------
def ai_model_sdk_path(instance, filename):
    base_dir = getattr(settings, 'AI_MODEL_SDK_DIR', None)
    return Path(base_dir, 'user_custom_sdk', f'{instance.project.hash}', f'ai_model_sdk_{instance.id}', filename)
    
class AIModelSDK(models.Model):
    name = models.CharField('AI Model SDK Name', max_length=128)
    project = models.ForeignKey(Project, verbose_name='Project', on_delete=models.CASCADE)
    
    ai_model_sdk_zip = models.FileField(upload_to=ai_model_sdk_path, max_length=512)
    ai_model_sdk_dir = models.CharField('', max_length=512, blank=True)
    ai_model_sdk_dir_offset = models.CharField('', max_length=512, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if (self.id is None):
            # --- dummy save, set id and zip files ---
            _tmp_ai_model_sdk_zip = self.ai_model_sdk_zip
            
            self.ai_model_sdk_zip = None
            super().save(*args, **kwargs)
            
            self.ai_model_sdk_zip = _tmp_ai_model_sdk_zip
            if ('force_insert' in kwargs):
                kwargs.pop('force_insert')
        
            # --- set ai_model_sdk_dir ---
            if (self.ai_model_sdk_dir is None):
                self.ai_model_sdk_dir = str(Path(
                                    getattr(settings, 'MEDIA_ROOT', None),
                                    getattr(settings, 'AI_MODEL_SDK_DIR', None),
                                    'user_custom_sdk',
                                    self.project.hash,
                                    f'ai_model_sdk_{self.id}'))
            if (self.ai_model_sdk_dir_offset is None):
                self.ai_model_sdk_dir_offset = str(Path(
                                    'user_custom_sdk',
                                    self.project.hash,
                                    f'ai_model_sdk_{self.id}'))
            
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
    ai_model_sdk = models.ForeignKey(AIModelSDK, verbose_name='AIModelSDK', on_delete=models.SET_NULL, null=True)
    
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


