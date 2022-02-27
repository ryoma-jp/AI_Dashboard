import os
import logging
import subprocess
import json
import hashlib

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
            # --- save database ---
            project = form.save(commit=False)
            project.hash = hashlib.sha256(project.name.encode()).hexdigest()
            project.save()
            
            # logging.info('-------------------------------------')
            # logging.info(project.hash)
            # logging.info('-------------------------------------')
            
            # --- create default dataset ---
            Dataset.objects.create(name='MNIST', project=project)
            Dataset.objects.create(name='CIFAR-10', project=project)
            
            # --- create project directory ---
            os.makedirs(os.path.join(settings.MEDIA_ROOT, settings.MODEL_DIR, project.hash))
            
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
            model.hash = hashlib.sha256(model.name.encode()).hexdigest()
            
            # --- create model directory ---
            project_dir = os.path.join(settings.MEDIA_ROOT, settings.MODEL_DIR, project.hash)
            model_dir = os.path.join(project_dir, model.hash)
            os.makedirs(model_dir)
            model.model_dir = model_dir
            
            # --- create environment directory ---
            env_dir = os.path.join(settings.ENV_DIR, project.hash, model.hash)
            os.makedirs(env_dir, exist_ok=True)
            
            # --- load config ---
            if (model.dataset.name == 'MNIST'):
                config_file = 'config_mnist.json'
            elif (model.dataset.name == 'CIFAR-10'):
                config_file = 'config_cifar10.json'
            else:
                config_file = 'config_blank.json'
            with open(os.path.join(settings.MEDIA_ROOT, settings.CONFIG_DIR, config_file), 'r') as f:
                dict_config = json.load(f)
            
            # --- set parameters ---
            dict_config['env']['fifo']['value'] = os.path.join(env_dir, 'fifo_trainer_ctl')
            dict_config['env']['result_dir']['value'] = model_dir
            dict_config['dataset']['dataset_dir']['value'] = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR)
            with open(os.path.join(model.model_dir, 'config.json'), 'w') as f:
                json.dump(dict_config, f, ensure_ascii=False, indent=4)
            
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

""" Function: model_paraemter_edit
 * edit model paramter
"""
def model_paraemter_edit(request, model_id):
    def _set_config_parameters(config, save_config_list):
        for key in save_config_list:
            if (request.POST[key] != ''):
                if (config[key]['dtype'] == 'int'):
                    config[key]['value'] = int(request.POST[key])
                elif (config[key]['dtype'] == 'float'):
                    config[key]['value'] = float(request.POST[key])
                elif (config[key]['dtype'] == 'bool'):
                    if (request.POST[key].lower() in ['true']):
                        config[key]['value'] = True
                    else:
                        config[key]['value'] = False
                else:
                    config[key]['value'] = request.POST[key]

    # --- load model(id=model_id) ---
    model = MlModel.objects.get(pk=model_id)
    
    # --- load config ---
    with open(os.path.join(model.model_dir, 'config.json'), 'r') as f:
        config_data = json.load(f)
    
    # logging.info('-------------------------------------')
    # logging.info(request)
    # logging.info(request.POST)
    # logging.info('-------------------------------------')
    
    if (request.method == 'POST'):
        if ('apply_parameters' in request.POST):
            # --- save dataset parameters ---
            save_config_list = [key for key in config_data['dataset'].keys() if ((key != 'data_augmentation') and (config_data['dataset'][key]['configurable']))]
            _set_config_parameters(config_data['dataset'], save_config_list)
            
            # --- save data augmentation parameters ---
            save_config_list = [key for key in config_data['dataset']['data_augmentation'].keys() if config_data['dataset']['data_augmentation'][key]['configurable']]
            _set_config_parameters(config_data['dataset']['data_augmentation'], save_config_list)
            
            # --- save model parameters ---
            save_config_list = [key for key in config_data['model'].keys() if config_data['model'][key]['configurable']]
            _set_config_parameters(config_data['model'], save_config_list)
            
            # --- save training parameters ---
            save_config_list = [key for key in config_data['training_parameter'].keys() if config_data['training_parameter'][key]['configurable']]
            _set_config_parameters(config_data['training_parameter'], save_config_list)
            
            with open(os.path.join(model.model_dir, 'config.json'), 'w') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=4)
        
        return redirect('model_paraemter_edit', model_id)
    else:
        text = get_version()
        context = {
            'config': config_data,
            'text': text
        }
        return render(request, 'model_parameter_edit.html', context)

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
    def _get_selected_object():
        project_name = request.session.get('training_view_selected_project', None)
        selected_project = Project.objects.get(name=project_name)
        
        model_name = request.session.get('training_view_selected_model', None)
        selected_model = MlModel.objects.get(name=model_name, project=selected_project)
        
        return selected_project, selected_model
    
    def _training_run():
        selected_project, selected_model = _get_selected_object()
        if (selected_model):
            logging.debug(selected_model)
            
            # --- Load config ---
            config_path = os.path.join(selected_model.model_dir, 'config.json')
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # --- Create FIFO ---
            fifo = config_data['env']['fifo']['value']
            if (not os.path.exists(fifo)):
                os.mkfifo(fifo)
            
            # --- Training Model ---
            main_path = os.path.abspath('./app/machine_learning/main.py')
            logging.debug(f'main_path: {main_path}')
            logging.debug(f'current working directory: {os.getcwd()}')
            subproc_training = subprocess.Popen(['python', main_path, '--config', config_path])
            logging.info(f'subproc: Training worker PID: {subproc_training.pid}')
            
            subproc_tensorboard = subprocess.Popen(['tensorboard', \
                                        '--logdir', selected_model.model_dir, \
                                        '--port', '6006'])
            logging.info(f'subproc: Tensorboard worker PID: {subproc_tensorboard.pid}')
            
            # --- Update status and Register PID to MlModel database ---
            selected_model.status = MlModel.STAT_TRAINING
            selected_model.training_pid = subproc_training.pid
            selected_model.tensorboard_pid = subproc_tensorboard.pid
            selected_model.save()
            
        return
    
    def _stop_trainer():
        selected_project, selected_model = _get_selected_object()
        if (selected_model):
            logging.debug(selected_model)
            
            # --- Load config ---
            config_path = os.path.join(selected_model.model_dir, 'config.json')
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # --- Get FIFO path ---
            fifo = config_data['env']['fifo']['value']
            
            # --- Send stop command ---
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

