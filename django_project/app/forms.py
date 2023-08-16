from django import forms
from app.models import Project, Dataset, MlModel, AIModelSDK

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ('name', 'description',)

class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ('name',
                  'train_zip',
                  'valid_zip',
                  'test_zip',
                  'meta_zip', )

class MlModelForm(forms.ModelForm):
    class Meta:
        model = MlModel
        fields = ('name', 'description',)

class AIModelSDKForm(forms.ModelForm):
    class Meta:
        model = AIModelSDK
        fields = ('name',
                  'ai_model_sdk_zip',
                )

