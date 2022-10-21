from django import forms
from app.models import Project, Dataset, MlModel

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

