import os
import logging
import psutil
import signal
import subprocess
import json

from pathlib import Path

from django.shortcuts import render, redirect

from app.models import Project, MlModel

from views_common import SidebarActiveStatus, get_version, get_all_fifo_command, get_jupyter_nb_url

# Create your views here.

def training(request):
    """ Function: training
     * training top
    """
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
            config_path = Path(selected_model.model_dir, 'config.json')
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # --- Training Model ---
            main_path = Path('./app/machine_learning/main.py').resolve()
            logging.debug(f'main_path: {main_path}')
            logging.debug(f'current working directory: {os.getcwd()}')
            subproc_training = subprocess.Popen(['python', main_path, '--mode', 'train', '--config', config_path])
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
            config_path = Path(selected_model.model_dir, 'config.json')
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
        logging.info('-------------------------------------')
        logging.info(model.model_dir)
        logging.info('-------------------------------------')
        
        config_path = Path(model.model_dir, 'config.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        logging.info('-------------------------------------')
        logging.info(model.tensorboard_pid)
        logging.info(psutil.pids())
        logging.info('-------------------------------------')
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
            
            logging.info('-------------------------------------')
            logging.info(psutil.pids())
            if (prev_model is not None):
                logging.info(prev_model.tensorboard_pid)
            else:
                logging.info("prev_model is None")
            if (curr_model is not None):
                logging.info(curr_model.tensorboard_pid)
            else:
                logging.info("curr_model is None")
            logging.info('-------------------------------------')
            # --- Close previous Tensorboard ---
            #  * https://psutil.readthedocs.io/en/latest/#kill-process-tree
            if ((prev_model is not None) and
                ((prev_model.project.hash+prev_model.hash) != (curr_model.project.hash+curr_model.hash)) and
                (prev_model.tensorboard_pid is not None) and
                (prev_model.tensorboard_pid in psutil.pids())):
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
        
        project = Project.objects.all().order_by('-id').reverse()
        
        # check for existence of selected project name
        project_name_list = [p.name for p in project]
        selected_project_name = request.session.get('training_view_selected_project', None)
        
        logging.info('-------------------------------------')
        logging.info(project_name_list)
        logging.info(selected_project_name)
        logging.info('-------------------------------------')
        
        if ((selected_project_name is not None) and (selected_project_name in project_name_list)):
            project_dropdown_selected = Project.objects.get(name=selected_project_name)
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
            config_path = Path(model_dropdown_selected.model_dir, 'config.json')
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
            'text': get_version(),
            'jupyter_nb_url': get_jupyter_nb_url(),
            'project_dropdown_selected': project_dropdown_selected,
            'model_dropdown_selected': model_dropdown_selected
        }
        return render(request, 'training.html', context)

def model_paraemter_edit(request, model_id):
    """ Function: model_paraemter_edit
     * edit model paramter
    """
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
    with open(Path(model.model_dir, 'config.json'), 'r') as f:
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
            
            with open(Path(model.model_dir, 'config.json'), 'w') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=4)
        
        return redirect('model_paraemter_edit', model_id)
    else:
        context = {
            'config': config_data,
            'text': get_version(),
            'jupyter_nb_url': get_jupyter_nb_url(),
        }
        return render(request, 'model_parameter_edit.html', context)

