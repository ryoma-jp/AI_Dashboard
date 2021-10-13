from django import forms
from app.models import DatasetFile, DatasetSelection

class DatasetFileForm(forms.ModelForm):
    class Meta:
        model = DatasetFile
        fields = ('x_train', 'y_train',
                  'x_valid', 'y_valid',
                  'x_test', 'y_test', )

class DatasetSelectionForm(forms.ModelForm):
    class Meta:
        model = DatasetSelection
        fields = ('selection', )
