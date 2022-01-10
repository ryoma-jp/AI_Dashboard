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
                  'train_zip', 'train_csv',
                  'valid_zip', 'valid_csv',
                  'test_zip', 'test_csv', )

class MlModelForm(forms.ModelForm):
    class Meta:
        model = MlModel
        fields = ('name', 'description',)

