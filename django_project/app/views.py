import os
import logging
import subprocess

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist

from app.models import Project, Dataset, MlModel
from app.forms import ProjectForm, DatasetForm, MlModelForm

from .machine_learning.lib.data_loader.data_loader import *
from .machine_learning.lib.trainer.trainer import *

# Create your views here.

""" Class: Sidebar active status
"""
class SidebarActiveStatus():
    def __init__(self):
        self.index = ''
        self.dataset = ''
        self.training = ''


""" Function: get_version
 * return version text
"""
def get_version():
    if (settings.DEBUG):
        return 'Debug mode'
    else:
        return '[T.B.D] VerX.XX'

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
        dataset = Dataset.objects.all()
        dataset_form = DatasetForm()
        models = MlModel.objects.all()
    else:
        project_form = ProjectForm()
        dataset = Dataset.objects.all()
        dataset_form = DatasetForm()
        models = MlModel.objects.all()
    
    sidebar_status = SidebarActiveStatus()
    sidebar_status.index = 'active'
    text = get_version()
    system_info = Trainer.GetSystemInfo()
    
    context = {
        'projects': projects,
        'project_form': project_form,
        'dataset': dataset,
        'dataset_form': dataset_form,
        'models': models,
        'sidebar_status': sidebar_status,
        'text': text,
        'system_info': system_info
    }
    return render(request, 'index.html', context)


""" Function: project_new
 * new project
"""
def project_new(request):
    if (request.method == 'POST'):
        form = ProjectForm(request.POST)
        if (form.is_valid()):
            project = form.save(commit=False)
            project.save()
            
            Dataset.objects.create(name='MNIST', project=project)
            Dataset.objects.create(name='CIFAR-10', project=project)
            
            return redirect('index')
    else:
        form = ProjectForm()
    
    text = get_version()
    
    context = {
        'form': form,
        'text': text,
    }
    return render(request, 'project_new.html', context)

""" Function: model_new
 * new model
"""
def model_new(request, project_id):
    # logging.info('-------------------------------------')
    # logging.info(request.method)
    # logging.info(request.POST)
    # logging.info('-------------------------------------')
    
    project = get_object_or_404(Project, pk=project_id)
    
    if (request.method == 'POST'):
        form = MlModelForm(request.POST)
        if (form.is_valid()):
            model = form.save(commit=False)
            model.project = project
            
            selected_model = request.POST.getlist('model_new_dataset_dropdown_submit')[0]
            model.dataset = get_object_or_404(Dataset.objects.filter(project=project, name=selected_model))
            
            model.status = model.STAT_IDLE
            model.save()
            
            return redirect('index')
    else:
        form = MlModelForm()
    
    model_new_dropdown_selected = None
    dataset = Dataset.objects.all().filter(project=project)
    text = get_version()
    
    context = {
        'model_new_dropdown_selected': model_new_dropdown_selected,
        'dataset': dataset,
        'form': form,
        'text': text
    }
    return render(request, 'model_new.html', context)

""" Function: dataset
 * dataset top
"""
def dataset(request):
    if (request.method == 'POST'):
        if ('dataset_view_dropdown' in request.POST):
            dropdown = request.POST.getlist('dataset_view_dropdown')
            for project in Project.objects.all():
                if (project.name in dropdown):
                    project.dataset_view_selected = 'checked'
                else:
                    project.dataset_view_selected = 'unchecked'
                project.save()
        return redirect('dataset')
    else:
        project = Project.objects.all()
        dataset = Dataset.objects.all()
        sidebar_status = SidebarActiveStatus()
        sidebar_status.dataset = 'active'
        text = get_version()
        
        dataset_view_dropdown_selected = None
        for project_ in project:
            if (project_.dataset_view_selected == 'checked'):
                dataset_view_dropdown_selected = project_
                break
        
        context = {
            'project': project,
            'dataset': dataset,
            'sidebar_status': sidebar_status,
            'text': text,
            'dataset_view_dropdown_selected': dataset_view_dropdown_selected
        }
        return render(request, 'dataset.html', context)

""" Function: training
 * training top
"""
def training(request):
    if (request.method == 'POST'):
        if ('training_view_project_dropdown' in request.POST):
            dropdown = request.POST.getlist('training_view_project_dropdown')
            for project in Project.objects.all():
                if (project.name in dropdown):
                    project.training_view_selected = 'checked'
                else:
                    project.training_view_selected = 'unchecked'
                project.save()
                
        if ('training_view_model_dropdown' in request.POST):
            dropdown = request.POST.getlist('training_view_model_dropdown')
            for model in MlModel.objects.all():
                if (model.name in dropdown):
                    model.training_view_selected = 'checked'
                else:
                    model.training_view_selected = 'unchecked'
                model.save()
                
        return redirect('training')
    else:
        sidebar_status = SidebarActiveStatus()
        sidebar_status.training = 'active'
        text = get_version()
        
        project = Project.objects.all()
        
        project_dropdown_selected = None
        for project_ in project:
            if (project_.training_view_selected == 'checked'):
                project_dropdown_selected = project_
        
        if (project_dropdown_selected):
            model = MlModel.objects.filter(project=project_dropdown_selected)
            model_dropdown_selected = None
            for model_ in model:
                if (model_.training_view_selected == 'checked'):
                    model_dropdown_selected = model_
        else:
            model = MlModel.objects.all()
            model_dropdown_selected = None
        
        context = {
            'project': project,
            'model': model,
            'sidebar_status': sidebar_status,
            'text': text,
            'project_dropdown_selected': project_dropdown_selected,
            'model_dropdown_selected': model_dropdown_selected
        }
        return render(request, 'training.html', context)

