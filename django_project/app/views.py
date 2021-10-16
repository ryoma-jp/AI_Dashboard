import os
import logging

from django.shortcuts import render
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from app.models import DatasetFile, DatasetSelection
from app.forms import DatasetFileForm, DatasetSelectionForm

from .machine_learning.lib.data_loader.data_loader import *
from .machine_learning.lib.trainer.trainer import *

# --- machine learning trainer ---
class MlTrainerStatus():
    IDLE = 0
    PREPARING = 1
    TRAINING = 2
    DONE = 3
ml_trainer = None
ml_trainer_status = MlTrainerStatus.IDLE

# Create your views here.

""" Function: index
 * show main view
"""
def index(request):
    def _proc_dataset_selection(request, model):
        if ('selection' in request.POST):
            if (model is not None):
                form = DatasetSelectionForm(request.POST, instance=model)
            else:
                form = DatasetSelectionForm(request.POST)
            
            if (form.is_valid()):
                logging.debug('form is valid')
                form.save()
        else:
            if (model is not None):
                form = DatasetSelectionForm(instance=model)
            else:
                form = DatasetSelectionForm()
        
        return form
    
    def _proc_dataset_upload(request):
        dataset_file_form = DatasetFileForm()
        logging.debug('dataset_file_form.fields: ')
        logging.debug(dataset_file_form.fields)
        if any((key in dataset_file_form.fields.keys()) for key in request.FILES.keys()):
            flg_update_dataset_file = True
        else:
            flg_update_dataset_file = False
        
        logging.debug('flg_update_dataset_file: ')
        logging.debug(flg_update_dataset_file)
        if (flg_update_dataset_file):
            files = request.FILES.keys()
            logging.debug('files: ')
            logging.debug(files)
            
            try:
                file = DatasetFile.objects.get(id=1)
            except ObjectDoesNotExist:
                file = DatasetFile()
            
            if ('train_zip' in files):
                dataset_file = request.FILES.getlist('train_zip', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.train_zip = dataset_file
        
            if ('train_csv' in files):
                dataset_file = request.FILES.getlist('train_csv', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.train_csv = dataset_file
        
            if ('valid_zip' in files):
                dataset_file = request.FILES.getlist('valid_zip', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.valid_zip = dataset_file
        
            if ('valid_csv' in files):
                dataset_file = request.FILES.getlist('valid_csv', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.valid_csv = dataset_file
        
            if ('test_zip' in files):
                dataset_file = request.FILES.getlist('test_zip', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.test_zip = dataset_file
        
            if ('test_csv' in files):
                dataset_file = request.FILES.getlist('test_csv', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.test_csv = dataset_file
            
            file.save()
        
        return dataset_file_form
    
    def _training_run(request):
        global ml_trainer_status
        logging.debug('training_run: ')
        logging.debug(request.POST.keys())
        if ('training_run' in request.POST.keys()):
            dataset_selection = DatasetSelection.objects.all()
            if (len(dataset_selection) > 0):
                logging.debug(dataset_selection[0].selection)
                if (ml_trainer_status == MlTrainerStatus.IDLE):
                    ml_trainer_status = MlTrainerStatus.PREPARING
                    train_parameters = {
                        'dataset_type': dataset_selection[0].selection,
                        'dataset_dir_root': os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR),
                        'train_zip': '',
                        'train_csv': '',
                        'valid_zip': '',
                        'valid_csv': '',
                        'test_zip': '',
                        'test_csv': '',
                        'model_dir': os.path.join(settings.MEDIA_ROOT, settings.MODEL_DIR),
                    }
                    
                    if (dataset_selection[0].selection == 'User data'):
                        dataset_file = DatasetFile.objects.all()
                        train_parameters['train_zip'] = os.path.basename(dataset_file[0].train_zip.name)
                        train_parameters['train_csv'] = os.path.basename(dataset_file[0].train_csv.name)
                        train_parameters['valid_zip'] = os.path.basename(dataset_file[0].valid_zip.name)
                        train_parameters['valid_csv'] = os.path.basename(dataset_file[0].valid_csv.name)
                        train_parameters['test_zip'] = os.path.basename(dataset_file[0].test_zip.name)
                        train_parameters['test_csv'] = os.path.basename(dataset_file[0].test_csv.name)
                    
                    logging.debug(train_parameters)
                    # --- Prepare Dataset ---
                    if (train_parameters['dataset_type'] == 'MNIST'):
                        logging.debug('Prepare dataset: MNIST')
                        dataset = DataLoaderMNIST(train_parameters['dataset_dir_root'], validation_split=0.2, one_hot=True, download=True)
                    elif (train_parameters['dataset_type'] == 'CIFAR-10'):
                        logging.debug('Prepare dataset: CIFAR-10')
                        dataset = DataLoaderCIFAR10(train_parameters['dataset_dir_root'], validation_split=0.2, one_hot=True, download=True)
                    elif (train_parameters['dataset_type'] == 'User data'):
                        # --- T.B.D ---
                        logging.debug('Prepare dataset: User data')
                        pass
                    else:
                        # --- Unknown dataset ---
                        logging.debug('[ERROR] Unknown dataset')
                        return
                    
                    # --- Training Model ---
                    ml_trainer_status = MlTrainerStatus.TRAINING
                    ml_trainer = TrainerCNN(dataset.train_images.shape[1:], output_dir=train_parameters['model_dir'],
                        optimizer="momentum", loss="categorical_crossentropy", initializer="he_normal")
                    logging.debug('Training Start')
                    logging.debug('Training Done')
        
        return
    
    def _reset_trainer(request):
        logging.debug('reset_trainer: ')
        logging.debug(request.POST.keys())
        if ('reset_trainer' in request.POST.keys()):
            ml_trainer.reset_status()
        
        return
    
#    logging.debug('request: ')
#    logging.debug(request.__dict__)
#
#    logging.debug('DatasetSelectionForm: ')
#    logging.debug(DatasetSelectionForm.__dict__)
#
#    logging.debug('DatasetSelection: ')
#    logging.debug(DatasetSelection.__dict__)

    try:
        dataset_selection = DatasetSelection.objects.get(pk=1)
    except ObjectDoesNotExist:
        dataset_selection = None
    logging.debug('dataset_selection: ')
    logging.debug(dataset_selection)
    
    if (request.method == 'POST'):
        logging.debug('request.POST: ')
        logging.debug(request.POST)
        logging.debug('request.FILES: ')
        logging.debug(request.FILES)
        
        # --- データセット選択プルダウン ---
        dataset_selection_form = _proc_dataset_selection(request, dataset_selection)
        dataset_selection = DatasetSelection.objects.get(pk=1)
        
        # --- データセットアップロードフォーム ---
        dataset_file_form = _proc_dataset_upload(request)
        
        # --- 学習実行 ---
        logging.debug('dataset_selection_form')
        logging.debug(dataset_selection_form)
        _training_run(request)
        
        # --- 状態のリセット ---
        _reset_trainer(request)
        
    else:
        dataset_file_form = DatasetFileForm()
        if (dataset_selection is not None):
            dataset_selection_form = DatasetSelectionForm(instance=dataset_selection)
        else:
            dataset_selection_form = DatasetSelectionForm()
    
    if (settings.DEBUG):
        text = 'Debug mode'
    else:
        text = '[T.B.D] VerX.XX'
    
    dataset_file = DatasetFile.objects.all()
    if (dataset_selection is None):
        dataset_selection = DatasetSelection()
    system_info = Trainer.GetSystemInfo()
    return render(request, 'index.html', {
               'dataset_file': dataset_file,
               'dataset_selection': dataset_selection,
               'dataset_file_form': dataset_file_form,
               'dataset_selection_form': dataset_selection_form,
               'text': text,
               'system_info': system_info})

