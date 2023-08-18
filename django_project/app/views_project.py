import os
import logging
import json
import pickle
import hashlib
import shutil
import psutil
import signal

from pathlib import Path

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings

from app.models import Project, Dataset, MlModel, AIModelSDK
from app.forms import ProjectForm, MlModelForm

from views_common import SidebarActiveStatus, get_version, load_dataset, get_jupyter_nb_url

# Create your views here.

def _create_project_hash(project):
    """ Function: _create_project_hash
     * create project hash
     * internal function of views_project
    """
    return hashlib.sha256(f'{project.id:08}'.encode()).hexdigest()

def _create_model_hash(project, model):
    """ Function: _create_model_hash
     * create model hash
     * internal function of views_project
    """
    return hashlib.sha256(f'{project.id:08}{model.id:08}'.encode()).hexdigest()

def _model_delete(request, model):
    """ Delete Model
    
    Delete model
    
    Args:
        request : request
        model : model object
    """
    
    # --- delete model data ---
    shutil.rmtree(model.model_dir)
    
    # --- delete tensorboard process ---
    if (model.tensorboard_pid is not None):
        if (psutil.pid_exists(model.tensorboard_pid)):
            p = psutil.Process(model.tensorboard_pid)
            c = p.children(recursive=True)
            c.append(p)
            for p in c:
                try:
                    p.send_signal(signal.SIGTERM)
                except psutil.NoSuchProcess:
                    pass
            gone, alive = psutil.wait_procs(c, timeout=3)
    
    # --- delete session data ---
    if 'training_view_selected_model' in request.session.keys():
        if (request.session['training_view_selected_model'] == model.name):
            del request.session['training_view_selected_model']
            request.session.modified = True
    
    if 'inference_view_selected_model' in request.session.keys():
        if (request.session['inference_view_selected_model'] == model.name):
            del request.session['inference_view_selected_model']
            request.session.modified = True
    
    # --- delete database ---
    model.delete()
    
    return

def _project_delete(request, project):
    """ Delete Project
    
    Delete project
    
    Args:
        request : request
        project : project object
    """
    
    # --- delete model data ---
    #   * trained data
    #   * tensorboard process
    #   * MlModel object
    for model in MlModel.objects.filter(project=project):
        _model_delete(request, model)
    
    # --- delete project data ---
    project_dir = Path(settings.MEDIA_ROOT, settings.MODEL_DIR, project.hash)
    if (project_dir.exists()):
        shutil.rmtree(project_dir)
    
    dataset_dir = Path(settings.MEDIA_ROOT, settings.DATASET_DIR, project.hash)
    if (dataset_dir.exists()):
        shutil.rmtree(dataset_dir)
    
    ai_model_sdk_dir = Path(settings.MEDIA_ROOT, settings.AI_MODEL_SDK_DIR, 'user_custom_sdk', project.hash)
    if (ai_model_sdk_dir.exists()):
        shutil.rmtree(ai_model_sdk_dir)
    
    # --- delete session data ---
    if 'training_view_selected_project' in request.session.keys():
        if (request.session['training_view_selected_project'] == project.name):
            del request.session['training_view_selected_project']
            request.session.modified = True
    
    if 'inference_view_selected_project' in request.session.keys():
        if (request.session['inference_view_selected_project'] == project.name):
            del request.session['inference_view_selected_project']
            request.session.modified = True
    
    # --- delete database ---
    project.delete()
    
    return


def project_new(request):
    """ Function: project_new
     * new project
    """
    if (request.method == 'POST'):
        form = ProjectForm(request.POST)
        if (form.is_valid()):
            # --- save database ---
            project = form.save(commit=True)       # commit=True: id確定のため
            project.hash = _create_project_hash(project)
            project.save()
            
            # logging.info('-------------------------------------')
            # logging.info(project.hash)
            # logging.info('-------------------------------------')
            
            # --- create default dataset ---
            Dataset.objects.create(name='MNIST', project=project, dataset_type=Dataset.DATASET_TYPE_IMAGE)
            Dataset.objects.create(name='CIFAR-10', project=project, dataset_type=Dataset.DATASET_TYPE_IMAGE)
            Dataset.objects.create(name='COCO2017', project=project, dataset_type=Dataset.DATASET_TYPE_IMAGE)
            Dataset.objects.create(name='PascalVOC2012', project=project, dataset_type=Dataset.DATASET_TYPE_IMAGE)
            Dataset.objects.create(name='CaliforniaHousing', project=project, dataset_type=Dataset.DATASET_TYPE_TABLE)
            
            # --- create sample SDK ---
            sample_sdk_path = Path(
                                   getattr(settings, 'MEDIA_ROOT', None),
                                   getattr(settings, 'AI_MODEL_SDK_DIR', None),
                                   'sample_sdk')
            AIModelSDK.objects.create(name='SimpleCNN', project=project, ai_model_sdk_dir=Path(sample_sdk_path, 'SimpleCNN'), ai_model_sdk_dir_offset=Path('sample_sdk', 'SimpleCNN'))

            # --- create project directory ---
            os.makedirs(Path(settings.MEDIA_ROOT, settings.MODEL_DIR, project.hash))
            
            return redirect('index')
    else:
        form = ProjectForm()
    
    context = {
        'form': form,
        'text': get_version(),
        'jupyter_nb_url': get_jupyter_nb_url(),
    }
    return render(request, 'project_new.html', context)

def project_edit(request, project_id):
    """ Function: project_edit
     * edit project
    """
    project = get_object_or_404(Project, pk=project_id)
    if (request.method == 'POST'):
        # logging.info('-------------------------------------')
        # logging.info(request.method)
        # logging.info(request.POST)
        # logging.info('-------------------------------------')
        
        if ('project_apply' in request.POST):
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
                
        elif ('project_delete' in request.POST):
            _project_delete(request, project)
            return redirect('index')
    else:
        initial_dict = dict(name=project.name, description=project.description)
        form = ProjectForm(initial=initial_dict)
    
    context = {
        'form': form,
        'text': get_version(),
        'jupyter_nb_url': get_jupyter_nb_url(),
    }
    return render(request, 'project_edit.html', context)

def model_new(request, project_id):
    """ Function: model_new
     * new model
    """
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
            selected_dataset = request.POST.getlist('model_new_dataset_dropdown_submit')[0]
            model.dataset = get_object_or_404(Dataset.objects.filter(project=project, name=selected_dataset))
            
            # --- dummy save ---
            model.save()       # commit=True: id確定のため
            
            # --- create hash ---
            model.hash = _create_model_hash(project, model)
            
            # --- create model directory ---
            project_dir = Path(settings.MEDIA_ROOT, settings.MODEL_DIR, project.hash)
            model_dir = Path(project_dir, model.hash)
            dataset_dir = Path(settings.MEDIA_ROOT, settings.DATASET_DIR, project.hash)
            os.makedirs(model_dir)
            model.model_dir = model_dir
            
            # --- create environment directory ---
            env_dir = Path(settings.ENV_DIR, project.hash, model.hash)
            os.makedirs(env_dir, exist_ok=True)
            
            # --- create dataset directory ---
            os.makedirs(dataset_dir, exist_ok=True)
            
            # --- load dataset ---
            dataset = load_dataset(model.dataset)
            
            # --- load config ---
            if (model.dataset.name == 'MNIST'):
                config_file = 'config_mnist.json'
            elif (model.dataset.name == 'CIFAR-10'):
                config_file = 'config_cifar10.json'
            elif (model.dataset.name == 'COCO2017'):
                config_file = 'config_coco2017.json'
            elif (model.dataset.name == 'PascalVOC2012'):
                config_file = 'config_pascal_voc2012.json'
            elif (model.dataset.name == 'CaliforniaHousing'):
                config_file = 'config_california_housing.json'
            else:
                config_file = 'config_blank.json'
            with open(Path(settings.MEDIA_ROOT, settings.CONFIG_DIR, config_file), 'r') as f:
                dict_config = json.load(f)
            
            # --- set parameters ---
            dict_config['env']['web_app_ctrl_fifo']['value'] = str(Path(env_dir, 'web_app_ctrl_fifo'))
            dict_config['env']['trainer_ctrl_fifo']['value'] = str(Path(env_dir, 'fifo_trainer_ctrl'))
            dict_config['env']['result_dir']['value'] = str(model.model_dir)
            #dict_config['dataset']['dataset_dir']['value'] = str(model.model_dir)	# directory that contains 'dataset.pkl'
            dict_config['dataset']['dataset_dir']['value'] = str(Path(dataset_dir, f'dataset_{model.dataset.id}'))
            
            selected_model = request.POST.getlist('model_new_model_dropdown_submit')[0]
            dict_config['model']['model_type']['value'] = selected_model
            model.ai_model_sdk = get_object_or_404(AIModelSDK.objects.filter(project=project, name=selected_model))
            
            # --- save config.json ---
            with open(Path(model.model_dir, 'config.json'), 'w') as f:
                json.dump(dict_config, f, ensure_ascii=False, indent=4)
                
            logging.info('-------------------------------------')
            logging.info(dict_config)
            logging.info('-------------------------------------')
            
            # --- save dataset ---
            model.dataset_pickle = Path(model.model_dir, 'dataset.pkl')
            with open(model.dataset_pickle, 'wb') as f:
                pickle.dump(dataset, f)
            
            # --- save database ---
            model.status = model.STAT_IDLE
            model.save()
            
            # --- Create trainer control FIFO ---
            fifo = dict_config['env']['trainer_ctrl_fifo']['value']
            if (not Path(fifo).exists()):
                os.mkfifo(fifo)
            
            # --- Create web app control FIFO ---
            fifo = dict_config['env']['web_app_ctrl_fifo']['value']
            if (not Path(fifo).exists()):
                os.mkfifo(fifo)
            
            return redirect('index')
    else:
        form = MlModelForm()
    
    model_new_dataset_selected = None
    dataset = Dataset.objects.all().filter(project=project).order_by('-id').reverse()
    
    model_new_model_selected = None
    
    ai_model_sdk = AIModelSDK.objects.all().filter(project=project).order_by('-id').reverse()

    context = {
        'model_new_model_selected': model_new_model_selected,
        'model_new_dataset_selected': model_new_dataset_selected,
        'dataset': dataset,
        'model': ai_model_sdk,
        'form': form,
        'text': get_version(),
        'jupyter_nb_url': get_jupyter_nb_url(),
    }
    return render(request, 'model_new.html', context)

def model_edit(request, project_id, model_id):
    """ Function: model_edit
     * edit model
    """
    # logging.info('-------------------------------------')
    # logging.info(request.method)
    # logging.info(request.POST)
    # logging.info('-------------------------------------')
    
    project = get_object_or_404(Project, pk=project_id)
    model = get_object_or_404(MlModel, pk=model_id, project=project)
    
    if (request.method == 'POST'):
        if ('model_apply' in request.POST):
            form = MlModelForm(request.POST)
            if (form.is_valid()):
                # --- get form data ---
                model.name = form.cleaned_data.get('name')
                model.description = form.cleaned_data.get('description')
                
                # --- get dataset object ---
                selected_dataset = request.POST.getlist('model_edit_dataset_dropdown_selected_submit')[0]
                model.dataset = get_object_or_404(Dataset.objects.filter(project=project, name=selected_dataset))
                
                # --- load config ---
                with open(Path(model.model_dir, 'config.json'), 'r') as f:
                    config_data = json.load(f)
                dataset_dir = config_data['dataset']['dataset_dir']['value']
                
                # --- preparing dataset ---
                dataset = load_dataset(model.dataset)
                model.dataset_pickle = Path(dataset_dir, 'dataset.pkl')
                with open(model.dataset_pickle, 'wb') as f:
                    pickle.dump(dataset, f)
                
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
                
        if ('model_delete' in request.POST):
            _model_delete(request, model)
            return redirect('index')
        
    else:
        initial_dict = dict(name=model.name, description=model.description)
        form = MlModelForm(initial=initial_dict)
    
    model_edit_dropdown_selected = model.dataset
    dataset = Dataset.objects.all().filter(project=project).order_by('-id').reverse()
    
    context = {
        'model_edit_dropdown_selected': model_edit_dropdown_selected,
        'dataset': dataset,
        'form': form,
        'text': get_version(),
        'jupyter_nb_url': get_jupyter_nb_url(),
    }
    return render(request, 'model_edit.html', context)

