import os
import fcntl
import logging
import subprocess
import json
import hashlib
import psutil
import signal

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

""" Function: get_recv_fifo_command
 * return recieved command
"""
def get_recv_fifo_command(fifo):
    fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    flags &= ~os.O_NONBLOCK
    fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    try:
        command = os.read(fd, 128)
        command = command.decode()[:-1]
        while (True):
            buf = os.read(fd, 65536)
            if not buf:
                break
    finally:
        os.close(fd)

    if (command):
        if (command == 'trainer_done'):
            logging.debug(f'{command} command recieved')
            return command
        else:
            logging.debug(f'Unknown command({command}) recieved')
            return None
    
    return None

""" Function: get_all_fifo_command
 * get all fifo command
"""
def get_all_fifo_command():
    projects = Project.objects.all().order_by('-id').reverse()
    for project in projects:
        models = MlModel.objects.filter(project=project).order_by('-id').reverse()
        
        for model in models:
            with open(os.path.join(model.model_dir, 'config.json'), 'r') as f:
                dict_config = json.load(f)
            
            while (True):
                recv_command = get_recv_fifo_command(dict_config['env']['web_app_ctrl_fifo']['value'])
                if (recv_command == 'trainer_done'):
                    model.training_pid = None
                    model.status = model.STAT_DONE
                    model.save()
                else:
                    break

""" Function: create_project_hash
 * create project hash
"""
def create_project_hash(project):
    return hashlib.sha256(f'{project.id:08}'.encode()).hexdigest()

""" Function: create_model_hash
 * create model hash
"""
def create_model_hash(project, model):
    return hashlib.sha256(f'{project.id:08}{model.id:08}'.encode()).hexdigest()


""" Function: index
 * show main view
"""
def index(request):
    get_all_fifo_command()
    
    projects = Project.objects.all().order_by('-id').reverse()
    project_form = ProjectForm()
    dataset = Dataset.objects.all().order_by('-id').reverse()
    dataset_form = DatasetForm()
    models = MlModel.objects.all().order_by('-id').reverse()
    
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
            project = form.save(commit=True)       # commit=True: id確定のため
            project.hash = create_project_hash(project)
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

""" Function: project_edit
 * edit project
"""
def project_edit(request, project_id):
    
    project = get_object_or_404(Project, pk=project_id)
    if (request.method == 'POST'):
        form = ProjectForm(request.POST)
        if (form.is_valid()):
            # --- get form data ---
            project.name = form.cleaned_data.get('name')
            project.description = form.cleaned_data.get('description')
            
            # logging.info('-------------------------------------')
            # logging.info(project.hash)
            # logging.info('-------------------------------------')
            
            # --- save database ---
            project.save()
            
            # --- clear session variables ---
            if 'training_view_selected_project' in request.session.keys():
                del request.session['training_view_selected_project']
                request.session.modified = True
            
            return redirect('index')
    else:
        initial_dict = dict(name=project.name, description=project.description)
        form = ProjectForm(initial=initial_dict)
    
    text = get_version()
    
    context = {
        'form': form,
        'text': text,
    }
    return render(request, 'project_edit.html', context)


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
            
            # --- dummy save ---
            model.save()       # commit=True: id確定のため
            
            # --- create hash ---
            model.hash = create_model_hash(project, model)
            
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
            dict_config['env']['web_app_ctrl_fifo']['value'] = os.path.join(env_dir, 'web_app_ctrl_fifo')
            dict_config['env']['trainer_ctrl_fifo']['value'] = os.path.join(env_dir, 'fifo_trainer_ctrl')
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
            
            # --- Create trainer control FIFO ---
            fifo = dict_config['env']['trainer_ctrl_fifo']['value']
            if (not os.path.exists(fifo)):
                os.mkfifo(fifo)
            
            # --- Create web app control FIFO ---
            fifo = dict_config['env']['web_app_ctrl_fifo']['value']
            if (not os.path.exists(fifo)):
                os.mkfifo(fifo)
            
            return redirect('index')
    else:
        form = MlModelForm()
    
    model_new_dropdown_selected = None
    dataset = Dataset.objects.all().filter(project=project).order_by('-id').reverse()
    text = get_version()
    
    context = {
        'model_new_dropdown_selected': model_new_dropdown_selected,
        'dataset': dataset,
        'form': form,
        'text': text
    }
    return render(request, 'model_new.html', context)

""" Function: model_edit
 * edit model
"""
def model_edit(request, project_id, model_id):
    # logging.info('-------------------------------------')
    # logging.info(request.method)
    # logging.info(request.POST)
    # logging.info('-------------------------------------')
    
    project = get_object_or_404(Project, pk=project_id)
    model = get_object_or_404(MlModel, pk=model_id, project=project)
    
    if (request.method == 'POST'):
        form = MlModelForm(request.POST)
        if (form.is_valid()):
            # --- get form data ---
            model.name = form.cleaned_data.get('name')
            model.description = form.cleaned_data.get('description')
            
            # --- get dataset object ---
            selected_dataset = request.POST.getlist('model_edit_dataset_dropdown_selected_submit')[0]
            model.dataset = get_object_or_404(Dataset.objects.filter(project=project, name=selected_dataset))
            
            # logging.info('-------------------------------------')
            # logging.info(dict_config)
            # logging.info('-------------------------------------')
            
            # --- save database ---
            model.save()
            
            # --- clear session variables ---
            if 'training_view_selected_model' in request.session.keys():
                del request.session['training_view_selected_model']
                request.session.modified = True
            
            return redirect('index')
    else:
        initial_dict = dict(name=model.name, description=model.description)
        form = MlModelForm(initial=initial_dict)
    
    model_edit_dropdown_selected = model.dataset
    dataset = Dataset.objects.all().filter(project=project).order_by('-id').reverse()
    text = get_version()
    
    context = {
        'model_edit_dropdown_selected': model_edit_dropdown_selected,
        'dataset': dataset,
        'form': form,
        'text': text
    }
    return render(request, 'model_edit.html', context)

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
            request.session['dataset_view_dropdown_selected_project'] = request.POST.getlist('dataset_view_dropdown')[0]
        return redirect('dataset')
    else:
        project = Project.objects.all().order_by('-id').reverse()
        dataset = Dataset.objects.all().order_by('-id').reverse()
        sidebar_status = SidebarActiveStatus()
        sidebar_status.dataset = 'active'
        text = get_version()
        
        project_name = request.session.get('dataset_view_dropdown_selected_project', None)
        if (project_name is not None):
            dropdown_selected_project = Project.objects.get(name=project_name)
        else:
            dropdown_selected_project = None
        
        # logging.info('-------------------------------------')
        # logging.info(project_name)
        # logging.info(dropdown_selected_project)
        # logging.info('-------------------------------------')
        context = {
            'project': project,
            'dataset': dataset,
            'sidebar_status': sidebar_status,
            'text': text,
            'dropdown_selected_project': dropdown_selected_project
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
            
            # --- Training Model ---
            main_path = os.path.abspath('./app/machine_learning/main.py')
            logging.debug(f'main_path: {main_path}')
            logging.debug(f'current working directory: {os.getcwd()}')
            subproc_training = subprocess.Popen(['python', main_path, '--config', config_path])
            logging.info(f'subproc: Training worker PID: {subproc_training.pid}')
            
            # --- Update status and Register PID to MlModel database ---
            selected_model.status = selected_model.STAT_TRAINING
            selected_model.training_pid = subproc_training.pid
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
            fifo = config_data['env']['trainer_ctrl_fifo']['value']
            
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
    
    def _launch_tensorboard(model):
        config_path = os.path.join(model.model_dir, 'config.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        if (not model.tensorboard_pid in psutil.pids()):
            subproc_tensorboard = subprocess.Popen(['tensorboard', \
                                        '--logdir', model.model_dir, \
                                        '--port', f'{config_data["env"]["tensorboard_port"]["value"]}'])
            logging.info(f'subproc: Tensorboard worker PID: {subproc_tensorboard.pid}')
            
            model.tensorboard_pid = subproc_tensorboard.pid
            model.save()
    
    # logging.info('-------------------------------------')
    # logging.info(request.method)
    # logging.info(request.POST)
    # logging.info('-------------------------------------')
    if (request.method == 'POST'):
        if ('training_view_project_dropdown' in request.POST):
            request.session['training_view_selected_project'] = request.POST.getlist('training_view_project_dropdown')[0]
                
        elif ('training_view_model_dropdown' in request.POST):
            curr_project = Project.objects.get(name=request.session['training_view_selected_project'])
            
            if 'training_view_selected_model' in request.session.keys():
                prev_model = MlModel.objects.get(name=request.session['training_view_selected_model'], project=curr_project)
            else:
                prev_model = None
            
            request.session['training_view_selected_model'] = request.POST.getlist('training_view_model_dropdown')[0]
            curr_model = MlModel.objects.get(name=request.session['training_view_selected_model'], project=curr_project)
            
            # --- Close previous Tensorboard ---
            #  * https://psutil.readthedocs.io/en/latest/#kill-process-tree
            if ((prev_model is not None) and (prev_model.tensorboard_pid is not None) and (prev_model.tensorboard_pid in psutil.pids())):
                p = psutil.Process(prev_model.tensorboard_pid)
                c = p.children(recursive=True)
                c.append(p)
                for p in c:
                    try:
                        p.send_signal(signal.SIGTERM)
                    except psutil.NoSuchProcess:
                        pass
                gone, alive = psutil.wait_procs(c, timeout=3)
                
                prev_model.tensorboard_pid = None
                prev_model.save()
            
            # --- Launch new Tensorboard ---
            _launch_tensorboard(curr_model)
            
        elif ('training_run' in request.POST):
            _training_run()
        
        elif ('stop_trainer' in request.POST):
            _stop_trainer()
        
        else:
            logging.warning('Unknown POST command:')
            logging.warning(request.POST)
        
        return redirect('training')
    else:
        get_all_fifo_command()
        sidebar_status = SidebarActiveStatus()
        sidebar_status.training = 'active'
        text = get_version()
        
        project = Project.objects.all().order_by('-id').reverse()
        project_name = request.session.get('training_view_selected_project', None)
        if (project_name is not None):
            project_dropdown_selected = Project.objects.get(name=project_name)
        else:
            project_dropdown_selected = None
        
        if (project_dropdown_selected):
            model = MlModel.objects.filter(project=project_dropdown_selected).order_by('-id').reverse()
            
            model_name = request.session.get('training_view_selected_model', None)
            if (model_name is not None):
                model_dropdown_selected = MlModel.objects.get(name=model_name, project=project_dropdown_selected)
                _launch_tensorboard(model_dropdown_selected)
            else:
                model_dropdown_selected = None
            
        else:
            model = MlModel.objects.all().order_by('-id').reverse()
            model_dropdown_selected = None
        
        # --- Get Tensorboard PORT ---
        if (model_dropdown_selected is not None):
            config_path = os.path.join(model_dropdown_selected.model_dir, 'config.json')
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            tensorboard_port = config_data["env"]["tensorboard_port"]["value"]
        else:
            tensorboard_port = None
        
        
        context = {
            'project': project,
            'model': model,
            'tensorboard_port': tensorboard_port,
            'sidebar_status': sidebar_status,
            'text': text,
            'project_dropdown_selected': project_dropdown_selected,
            'model_dropdown_selected': model_dropdown_selected
        }
        return render(request, 'training.html', context)

