import os
import logging
import subprocess
import json

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
    
    projects = Project.objects.all()
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
            
            # --- get dataset object ---
            selected_model = request.POST.getlist('model_new_dataset_dropdown_submit')[0]
            model.dataset = get_object_or_404(Dataset.objects.filter(project=project, name=selected_model))
            
            # --- load config ---
            if (model.dataset.name == 'MNIST'):
                config_file = 'config_mnist.json'
            elif (model.dataset.name == 'CIFAR-10'):
                config_file = 'config_cifar10.json'
            else:
                config_file = 'config_blank.json'
            with open(os.path.join(settings.MEDIA_ROOT, settings.CONFIG_DIR, config_file), 'r') as f:
                dict_config = json.load(f)
            # logging.info('-------------------------------------')
            # logging.info(dict_config)
            # logging.info('-------------------------------------')
            
            # --- save database ---
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
            request.session['dataset_view_dropdown_selected'] = request.POST.getlist('dataset_view_dropdown')[0]
        return redirect('dataset')
    else:
        project = Project.objects.all()
        dataset = Dataset.objects.all()
        sidebar_status = SidebarActiveStatus()
        sidebar_status.dataset = 'active'
        text = get_version()
        
        project_name = request.session.get('dataset_view_dropdown_selected', None)
        if (project_name is not None):
            dataset_view_dropdown_selected = Project.objects.get(name=project_name)
        else:
            dataset_view_dropdown_selected = None
        
        # logging.info('-------------------------------------')
        # logging.info(project_name)
        # logging.info(dataset_view_dropdown_selected)
        # logging.info('-------------------------------------')
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
    def _training_run():
        project_name = request.session.get('training_view_selected_project', None)
        selected_project = Project.objects.get(name=project_name)
        
        model_name = request.session.get('training_view_selected_model', None)
        selected_model = MlModel.objects.get(name=model_name, project=selected_project)
        
        if (selected_model):
            logging.debug(selected_model)
                
            # --- Create FIFO ---
            fifo = '/tmp/fifo_trainer_ctl'
            if (not os.path.exists(fifo)):
                os.mkfifo(fifo)
            
            # --- Training Model ---
            train_parameters = {
                'dataset_type': selected_model.dataset.name,
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
            subproc_training = subprocess.Popen(['python', main_path, \
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
                                        '--epochs', '400', \
                                        '--result_dir', train_parameters['model_dir']])
            logging.info(f'subproc: Training worker PID: {subproc_training.pid}')
            
            subproc_tensorboard = subprocess.Popen(['tensorboard', \
                                        '--logdir', train_parameters['model_dir'], \
                                        '--port', '6006'])
            logging.info(f'subproc: Tensorboard worker PID: {subproc_tensorboard.pid}')
            
            # --- Update status and Register PID to MlModel database ---
            selected_model.status = MlModel.STAT_TRAINING
            selected_model.training_pid = subproc_training.pid
            selected_model.tensorboard_pid = subproc_tensorboard.pid
            selected_model.save()
            
        return
    
    def _stop_trainer():
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
    
    # logging.info('-------------------------------------')
    # logging.info(request.method)
    # logging.info(request.POST)
    # logging.info('-------------------------------------')
    if (request.method == 'POST'):
        if ('training_view_project_dropdown' in request.POST):
            request.session['training_view_selected_project'] = request.POST.getlist('training_view_project_dropdown')[0]
                
        elif ('training_view_model_dropdown' in request.POST):
            request.session['training_view_selected_model'] = request.POST.getlist('training_view_model_dropdown')[0]
                
        elif ('training_run' in request.POST):
            _training_run()
        
        elif ('stop_trainer' in request.POST):
            _stop_trainer()
        
        else:
            logging.warning('Unknown POST command:')
            logging.warning(request.POST)
        
        return redirect('training')
    else:
        sidebar_status = SidebarActiveStatus()
        sidebar_status.training = 'active'
        text = get_version()
        
        project = Project.objects.all()
        project_name = request.session.get('training_view_selected_project', None)
        if (project_name is not None):
            project_dropdown_selected = Project.objects.get(name=project_name)
        else:
            project_dropdown_selected = None
        
        if (project_dropdown_selected):
            model = MlModel.objects.filter(project=project_dropdown_selected)
            
            model_name = request.session.get('training_view_selected_model', None)
            if (model_name is not None):
                model_dropdown_selected = MlModel.objects.get(name=model_name, project=project_dropdown_selected)
            else:
                model_dropdown_selected = None
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

