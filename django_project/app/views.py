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
g_ml_trainer = {
    'ml_trainer': None,
    'ml_trainer_status': MlTrainerStatus.IDLE
}

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
    
    def _training_run(request, trainer):
        logging.debug('training_run: ')
        logging.debug(request.POST.keys())
        if ('training_run' in request.POST.keys()):
            dataset_selection = DatasetSelection.objects.all()
            if (len(dataset_selection) > 0):
                logging.debug(dataset_selection[0].selection)
                if (trainer['ml_trainer_status'] == MlTrainerStatus.IDLE):
                    trainer['ml_trainer_status'] = MlTrainerStatus.PREPARING
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
                    x_train, x_val, x_test = dataset.normalization("max")
                    y_train = dataset.train_labels
                    y_val = dataset.validation_labels
                    y_test = dataset.test_labels
                    data_augmentation = {
                        'rotation_range': 5,
                        'width_shift_range': 0.2,
                        'height_shift_range': 0.2,
                        'zoom_range': 0.2,
                        'channel_shift_range': 0.2,
                        'horizontal_flip': True,
                    }
                    
                    # --- Training Model ---
                    trainer['ml_trainer_status'] = MlTrainerStatus.TRAINING
                    trainer['ml_trainer'] = TrainerCNN(dataset.train_images.shape[1:], output_dir=train_parameters['model_dir'],
                        optimizer="momentum", loss="categorical_crossentropy", initializer="he_normal")
                    logging.debug('Training Start')
                    trainer['ml_trainer'].fit(x_train, y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test,
                        batch_size=100, da_params=data_augmentation, epochs=10)
                    trainer['ml_trainer'].save_model()
                    
                    trainer['ml_trainer_status'] = MlTrainerStatus.DONE
                    logging.debug('Training Done')
                    
        return
    
    def _suspend_trainer(request, trainer):
        logging.debug('suspend_trainer: ')
        logging.debug(request.POST.keys())
        if ('suspend_trainer' in request.POST.keys()):
            trainer['ml_trainer'].suspend()
        
        return
    
    def _reset_trainer(request, trainer):
        logging.debug('reset_trainer: ')
        logging.debug(request.POST.keys())
        if ('reset_trainer' in request.POST.keys()):
            if (trainer['ml_trainer_status'] == MlTrainerStatus.DONE):
                trainer['ml_trainer'].release_memory()
                del trainer['ml_trainer']
                trainer['ml_trainer'] = None
                trainer['ml_trainer_status'] = MlTrainerStatus.IDLE
        
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
        dataset_selection = DatasetSelection()
        dataset_selection.save()
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
        logging.debug('g_ml_trainer (before _training_run())')
        logging.debug(g_ml_trainer)
        _training_run(request, g_ml_trainer)
        logging.debug('g_ml_trainer (after _training_run())')
        logging.debug(g_ml_trainer)
        
        # --- 学習中断 ---
        _suspend_trainer(request, g_ml_trainer)
        
        # --- 状態のリセット ---
        logging.debug('g_ml_trainer (before _reset_trainer())')
        logging.debug(g_ml_trainer)
        _reset_trainer(request, g_ml_trainer)
        logging.debug('g_ml_trainer (after _reset_trainer())')
        logging.debug(g_ml_trainer)
        
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

