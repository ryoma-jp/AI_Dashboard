from django import forms
from app.models import DatasetFile, DatasetSelection

class DatasetFileForm(forms.ModelForm):
    class Meta:
        model = DatasetFile
        fields = ('train_zip', 'train_csv',
                  'valid_zip', 'valid_csv',
                  'test_zip', 'test_csv', )

class DatasetSelectionForm(forms.ModelForm):
    class Meta:
        model = DatasetSelection
        fields = ('selection', )
