import os
import logging
import subprocess

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist

from app.models import Project, CustomDataset, MlModel
from app.forms import ProjectForm, CustomDatasetForm, MlModelForm

from .machine_learning.lib.data_loader.data_loader import *
from .machine_learning.lib.trainer.trainer import *

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
        logging.debug('training_run: ')
        logging.debug(request.POST.keys())
        if ('training_run' in request.POST.keys()):
            dataset_selection = DatasetSelection.objects.all()
            if (len(dataset_selection) > 0):
                logging.debug(dataset_selection[0].selection)
                    
                # --- Create FIFO ---
                fifo = '/tmp/fifo_trainer_ctl'
                if (not os.path.exists(fifo)):
                    os.mkfifo(fifo)
                
                # --- Training Model ---
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
                
                main_path = os.path.abspath('./app/machine_learning/main.py')
                logging.debug(f'main_path: {main_path}')
                logging.debug(f'current working directory: {os.getcwd()}')
                subproc = subprocess.Popen(['python', main_path, \
                                            '--fifo', fifo, \
                                            '--data_type', train_parameters['dataset_type'], \
                                            '--dataset_dir', train_parameters['dataset_dir_root'], \
                                            '--model_type', 'SimpleCNN', \
                                            '--data_augmentation', '5,0.2,0.2,0.2,0.2,True', \
                                            '--optimizer', 'momentum', \
                                            '--batch_size', '100', \
                                            '--initializer', 'he_normal', \
                                            '--dropout_rate', '0.25', \
                                            '--loss_func', 'categorical_crossentropy', \
                                            '--epochs', '10', \
                                            '--result_dir', train_parameters['model_dir']])
                logging.debug(f'subproc PID: {subproc.pid}')
                logging.debug('Training Done')
                    
        return
    
    def _suspend_trainer(request):
        logging.debug('suspend_trainer: ')
        logging.debug(request.POST.keys())
        if ('suspend_trainer' in request.POST.keys()):
            fifo = '/tmp/fifo_trainer_ctl'
            with open(fifo, 'w') as f:
                f.write('stop\n')
        
        return
    
    def _reset_trainer(request):
        logging.debug('reset_trainer: ')
        logging.debug(request.POST.keys())
        '''
        if ('reset_trainer' in request.POST.keys()):
            trainer.release_memory()
        '''
        return
    
    
    projects = Project.objects.all()
    if (projects):
        '''
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
            _training_run(request)
            
            # --- 学習中断 ---
            _suspend_trainer(request)
            
            # --- 状態のリセット ---
            _reset_trainer(request)
            
        else:
            dataset_file_form = DatasetFileForm()
            if (dataset_selection is not None):
                dataset_selection_form = DatasetSelectionForm(instance=dataset_selection)
            else:
                dataset_selection_form = DatasetSelectionForm()
        
        dataset_file = DatasetFile.objects.all()
        if (dataset_selection is None):
            dataset_selection = DatasetSelection()
        '''
        project_form = ProjectForm()
        custom_dataset = CustomDataset.objects.all()
        custom_dataset_form = CustomDatasetForm()
        models = MlModel.objects.all()
    else:
        project_form = ProjectForm()
        custom_dataset = CustomDataset.objects.all()
        custom_dataset_form = CustomDatasetForm()
        models = MlModel.objects.all()
    
    if (settings.DEBUG):
        text = 'Debug mode'
    else:
        text = '[T.B.D] VerX.XX'
    system_info = Trainer.GetSystemInfo()
    return render(request, 'index.html', {
               'projects': projects,
               'project_form': project_form,
               'custom_dataset': custom_dataset,
               'custom_dataset_form': custom_dataset_form,
               'models': models,
               'text': text,
               'system_info': system_info})


""" Function: project_new
 * new project
"""
def project_new(request):
    if (request.method == 'POST'):
        form = ProjectForm(request.POST)
        if (form.is_valid()):
            project = form.save(commit=False)
            project.save()
            
            return redirect('index')
    else:
        form = ProjectForm()
    
    return render(request, 'project_new.html', {'form': form})

""" Function: model_new
 * new model
"""
def model_new(request, project_id):
    project = get_object_or_404(Project, pk=project_id)
    
    if (request.method == 'POST'):
        form = MlModelForm(request.POST)
        if (form.is_valid()):
            model = form.save(commit=False)
            model.project = project
            model.save()
            
            return redirect('index')
    else:
        form = MlModelForm()
    
    return render(request, 'model_new.html', {'form': form})

