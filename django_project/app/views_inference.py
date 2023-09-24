import sys
import os
import pickle
import json
import subprocess
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from collections import OrderedDict

from django.shortcuts import render, redirect

from tensorflow.keras.utils import to_categorical

from app.models import Project, MlModel, Dataset
from machine_learning.lib.utils.utils import JsonEncoder
from machine_learning.lib.data_loader.data_loader import load_dataset_from_tfrecord

from views_common import SidebarActiveStatus, get_version, get_jupyter_nb_url, get_dataloader_obj
from django.http import FileResponse

# Create your views here.

def inference(request):
    """ Function: inference
     * inference top
    """
    def _get_selected_object():
        project_name = request.session.get('inference_view_selected_project', None)
        selected_project = Project.objects.get(name=project_name)
        
        model_name = request.session.get('inference_view_selected_model', None)
        selected_model = MlModel.objects.get(name=model_name, project=selected_project)
        
        return selected_project, selected_model
    
    def _inference_run():
        selected_project, selected_model = _get_selected_object()
        if (selected_model):
            logging.debug(selected_model)
            
            # --- Load config ---
            config_path = Path(selected_model.model_dir, 'config.json')
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # --- Predict ---
            if False:
                main_path = Path('./app/machine_learning/main.py').resolve()
                logging.info(f'main_path: {main_path}')
                logging.info(f'current working directory: {os.getcwd()}')
                subproc_inference = subprocess.Popen(['python', main_path, '--mode', 'predict', '--config', config_path])
            else:
                # --- add AI Model SDK path to Python path ---
                sys.path.append(selected_model.ai_model_sdk.ai_model_sdk_dir)

                # --- import AI Model SDK ---
                from ai_model_sdk import AI_Model_SDK
                logging.info(AI_Model_SDK.__version__)

                # --- Prepare inference ---
                dataset = Path(config_data['dataset']['dataset_dir']['value'], 'dataset.pkl')
                model_params = {
                    'model_path': selected_model.model_dir,
                }
                ai_model_sdk = AI_Model_SDK(dataset, model_params)

                ai_model_sdk.load_dataset()
                trained_model = Path(selected_model.model_dir, 'models')
                ai_model_sdk.load_model(trained_model)

                task_table = {
                    'img_clf': 'classification',
                }
                with open(dataset, 'rb') as f:
                    dataset = pickle.load(f)
                train_dataset = load_dataset_from_tfrecord(
                    task_table[config_data['inference_parameter']['model']['task']['value']],
                    dataset.train_dataset['tfrecord_path'], 
                    dataset.train_dataset['class_name_file_path'],
                    dataset.train_dataset['model_input_size'])
                #train_dataset = train_dataset.batch(ai_model_sdk.batch_size)
                validation_dataset = load_dataset_from_tfrecord(
                    task_table[config_data['inference_parameter']['model']['task']['value']],
                    dataset.validation_dataset['tfrecord_path'], 
                    dataset.validation_dataset['class_name_file_path'],
                    dataset.validation_dataset['model_input_size'])
                #validation_dataset = validation_dataset.batch(ai_model_sdk.batch_size)
                test_dataset = load_dataset_from_tfrecord(
                    task_table[config_data['inference_parameter']['model']['task']['value']],
                    dataset.test_dataset['tfrecord_path'], 
                    dataset.test_dataset['class_name_file_path'],
                    dataset.test_dataset['model_input_size'])
                #test_dataset = test_dataset.batch(ai_model_sdk.batch_size)
                dict_evaluations = {}
                
                input_tensor = []
                target_tensor = []
                for train_batch in train_dataset:
                    input_tensor.append(train_batch[0].numpy().tolist())
                    target_tensor.append(train_batch[1].numpy().tolist())
                input_tensor = np.array(input_tensor, dtype=np.float32)
                target_tensor = to_categorical(np.array(target_tensor, dtype=int))
                prediction = ai_model_sdk.predict(input_tensor, preprocessing=True)

                # --- save prediction ---
                #  - np.argmax is tentative
                json_data = []
                logging.info(f'target_tensor: {target_tensor}')
                for id, pred, target in zip(np.arange(0, len(target_tensor)), prediction, target_tensor):
                    json_data.append({
                        'id': id,
                        'prediction': np.argmax(pred),
                        'target': np.argmax(target),
                    })
                evaluation_dir = Path(selected_model.model_dir, 'evaluations')
                os.makedirs(evaluation_dir, exist_ok=True)
                with open(Path(evaluation_dir, 'train_prediction.json'), 'w') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4, cls=JsonEncoder)
                pd.DataFrame(json_data).to_csv(Path(evaluation_dir, 'train_prediction.csv'), index=False)

                # --- evaluation ---
                scores = ai_model_sdk.eval_model(prediction, target_tensor)
                for key in scores.keys():
                    dict_evaluations[f'train {key}'] = scores[key]

                input_tensor = []
                target_tensor = []
                for validation_batch in validation_dataset:
                    input_tensor.append(validation_batch[0].numpy().tolist())
                    target_tensor.append(validation_batch[1].numpy().tolist())
                input_tensor = np.array(input_tensor, dtype=np.float32)
                target_tensor = to_categorical(np.array(target_tensor, dtype=int))
                prediction = ai_model_sdk.predict(input_tensor, preprocessing=True)

                # --- save prediction ---
                #  - np.argmax is tentative
                json_data = []
                logging.info(f'target_tensor: {target_tensor}')
                for id, pred, target in zip(np.arange(0, len(target_tensor)), prediction, target_tensor):
                    json_data.append({
                        'id': id,
                        'prediction': np.argmax(pred),
                        'target': np.argmax(target),
                    })
                evaluation_dir = Path(selected_model.model_dir, 'evaluations')
                os.makedirs(evaluation_dir, exist_ok=True)
                with open(Path(evaluation_dir, 'validation_prediction.json'), 'w') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4, cls=JsonEncoder)
                pd.DataFrame(json_data).to_csv(Path(evaluation_dir, 'validation_prediction.csv'), index=False)

                # --- evaluation ---
                scores = ai_model_sdk.eval_model(prediction, target_tensor)
                for key in scores.keys():
                    dict_evaluations[f'validation {key}'] = scores[key]

                input_tensor = []
                target_tensor = []
                for test_batch in test_dataset:
                    input_tensor.append(test_batch[0].numpy().tolist())
                    target_tensor.append(test_batch[1].numpy().tolist())
                input_tensor = np.array(input_tensor, dtype=np.float32)
                target_tensor = to_categorical(np.array(target_tensor, dtype=int))
                prediction = ai_model_sdk.predict(input_tensor, preprocessing=True)

                # --- save prediction ---
                #  - np.argmax is tentative
                json_data = []
                logging.info(f'target_tensor: {target_tensor}')
                for id, pred, target in zip(np.arange(0, len(target_tensor)), prediction, target_tensor):
                    json_data.append({
                        'id': id,
                        'prediction': np.argmax(pred),
                        'target': np.argmax(target),
                    })
                evaluation_dir = Path(selected_model.model_dir, 'evaluations')
                os.makedirs(evaluation_dir, exist_ok=True)
                with open(Path(evaluation_dir, 'test_prediction.json'), 'w') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4, cls=JsonEncoder)
                pd.DataFrame(json_data).to_csv(Path(evaluation_dir, 'test_prediction.csv'), index=False)

                # --- evaluation ---
                scores = ai_model_sdk.eval_model(prediction, target_tensor)
                for key in scores.keys():
                    dict_evaluations[f'test {key}'] = scores[key]

                ## --- Dataset loop ---
                #dataset_list = ['train', 'validation', 'test']
                #dict_evaluations = {}
                #for dataset_name in dataset_list:
                #    # --- Create instance ---
                #    dataset = Path(config_data['dataset']['dataset_dir']['value'], 'dataset.pkl')
                #    dataset_path = config_data['dataset']['dataset_dir']['value']
                #    dataset_params = {
                #        'meta': Path(dataset_path, 'meta', 'info.json'),
                #        'inference': Path(dataset_path, dataset_name, 'info.json'),
                #    }
                #    model_params = {
                #        'model_path': selected_model.model_dir,
                #    }
                #    ai_model_sdk = AI_Model_SDK(dataset, model_params)
#
                #    # --- load dataset ---
                #    ai_model_sdk.load_dataset()
#
                #    # --- load model ---
                #    trained_model = Path(selected_model.model_dir, 'models')
                #    ai_model_sdk.load_model(trained_model)
#
                #    # --- inference ---
                #    prediction = ai_model_sdk.predict(ai_model_sdk.x_inference, preprocessing=False)
                #    logging.info(prediction.shape)
                #    logging.info(prediction)
#
                #    # --- save prediction ---
                #    #  - np.argmax is tentative
                #    json_data = []
                #    if (ai_model_sdk.y_inference is None):
                #        for id, pred in zip(ai_model_sdk.y_inference_info['id'], prediction):
                #            json_data.append({
                #                'id': id,
                #                'prediction': np.argmax(pred),
                #                'target': '(no data)',
                #            })
                #    else:
                #        logging.info(f'ai_model_sdk.y_inference: {ai_model_sdk.y_inference}')
                #        for id, pred, target in zip(ai_model_sdk.y_inference_info['id'], prediction, ai_model_sdk.y_inference):
                #            json_data.append({
                #                'id': id,
                #                'prediction': np.argmax(pred),
                #                'target': np.argmax(target),
                #            })
                #    evaluation_dir = Path(selected_model.model_dir, 'evaluations')
                #    os.makedirs(evaluation_dir, exist_ok=True)
                #    with open(Path(evaluation_dir, f'{dataset_name}_prediction.json'), 'w') as f:
                #        json.dump(json_data, f, ensure_ascii=False, indent=4, cls=JsonEncoder)
                #    pd.DataFrame(json_data).to_csv(Path(evaluation_dir, f'{dataset_name}_prediction.csv'), index=False)
#
                #    # --- evaluation ---
                #    scores = ai_model_sdk.eval_model(prediction, ai_model_sdk.y_inference)
                #    for key in scores.keys():
                #        dict_evaluations[f'{dataset_name} {key}'] = scores[key]

                # --- save evaluation ---
                with open(Path(evaluation_dir, f'evaluations.json'), 'w') as f:
                    json.dump(dict_evaluations, f, ensure_ascii=False, indent=4, cls=JsonEncoder)

                # --- unimport AI Model SDK ---
                del AI_Model_SDK

                # --- remove AI Model SDK path from Python path ---
                sys.path.remove(selected_model.ai_model_sdk.ai_model_sdk_dir)

    # logging.info('-------------------------------------')
    # logging.info(request.method)
    # logging.info(request.POST)
    # logging.info('-------------------------------------')
    if (request.method == 'POST'):
        if ('inference_view_project_dropdown' in request.POST):
            request.session['inference_view_selected_project'] = request.POST.getlist('inference_view_project_dropdown')[0]
                
        elif ('inference_view_model_dropdown' in request.POST):
            request.session['inference_view_selected_model'] = request.POST.getlist('inference_view_model_dropdown')[0]
            
        elif ('inference_view_dataset_dropdown' in request.POST):
            pass
            # (T.B.D)
            #   * dataset dropdown will be selected dataset that user required to inference
            
            # for debug
            # request.session['inference_view_selected_dataset'] = request.POST.getlist('inference_view_dataset_dropdown')[0]
            # curr_project = Project.objects.get(name=request.session['inference_view_selected_project'])
            # curr_dataset = Dataset.objects.get(name=request.session['inference_view_selected_dataset'], project=curr_project)
            
        elif ('inference_run' in request.POST):
            _inference_run()
        
        elif ('prediction_filter' in request.POST):
            request.session['prediction_filter'] = request.POST.getlist('prediction_filter')[0]
        
        elif ('prediction_data_type' in request.POST):
            request.session['prediction_data_type'] = request.POST.getlist('prediction_data_type')[0]
        
        else:
            logging.warning('Unknown POST command:')
            logging.warning(request.POST)
        
        return redirect('inference')
    else:
        sidebar_status = SidebarActiveStatus()
        sidebar_status.inference = 'active'
        
        project = Project.objects.all().order_by('-id').reverse()
        
        # check for existence of selected project name
        project_name_list = [p.name for p in project]
        selected_project_name = request.session.get('inference_view_selected_project', None)
        
        logging.info('-------------------------------------')
        logging.info(project_name_list)
        logging.info(selected_project_name)
        logging.info('-------------------------------------')
        
        if ((selected_project_name is not None) and (selected_project_name in project_name_list)):
            project_dropdown_selected = Project.objects.get(name=selected_project_name)
        else:
            project_dropdown_selected = None
        
        if (project_dropdown_selected):
            # --- get model list and selected model ---
            model = MlModel.objects.filter(project=project_dropdown_selected).order_by('-id').reverse()
            
            model_name = request.session.get('inference_view_selected_model', None)
            if (model_name in [f.name for f in MlModel.objects.filter(project=project_dropdown_selected)]):
                model_dropdown_selected = MlModel.objects.get(name=model_name, project=project_dropdown_selected)
            else:
                model_dropdown_selected = None
            
            # --- get dataset list and selected dataset (T.B.D) ---
            dataset = Dataset.objects.filter(project=project_dropdown_selected).order_by('-id').reverse()
            if (model_dropdown_selected is not None):
                dataset_dropdown_selected = model_dropdown_selected.dataset
            else:
                dataset_dropdown_selected = None
            
            #
            #dataset_name_list = [d.name for d in Dataset.objects.all().order_by('-id')]
            #selected_dataset_name = request.session.get('inference_view_selected_dataset', None)
            #logging.info('-------------------------------------')
            #logging.info(dataset_name_list)
            #logging.info(selected_dataset_name)
            #logging.info('-------------------------------------')
            #if ((selected_dataset_name is not None) and (selected_dataset_name in dataset_name_list)):
            #    dataset_dropdown_selected = Dataset.objects.get(name=selected_dataset_name, project=project_dropdown_selected)
            #else:
            #    dataset_dropdown_selected = None
            
            
        else:
            model = MlModel.objects.all().order_by('-id').reverse()
            model_dropdown_selected = None
        
            dataset = Dataset.objects.all().order_by('-id').reverse()
            dataset_dropdown_selected = None
        
        # --- Check prediction filter ---
        prediction_filter_selected = request.session.get('prediction_filter', 'All')
        
        # --- Check prediction data type
        prediction_data_type_selected = request.session.get('prediction_data_type', 'Test')
        
        # --- Load DataLoader object and prediction ---
        if (dataset_dropdown_selected is not None):
            # --- get DataLoader object ---
            dataloader_obj = get_dataloader_obj(dataset_dropdown_selected)
            
            # --- get prediction ---
            prediction_json = Path(model_dropdown_selected.model_dir, 'evaluations', f'{prediction_data_type_selected.lower()}_prediction.json')
            if (prediction_json.exists()):
                with open(prediction_json, 'r') as f:
                    prediction = json.load(f)
            else:
                prediction = None
        else:
            dataloader_obj = None
            prediction = None
        
        
        context = {
            'project': project,
            'model': model,
            'dataset': dataset,
            'sidebar_status': sidebar_status,
            'text': get_version(),
            'jupyter_nb_url': get_jupyter_nb_url(),
            'project_dropdown_selected': project_dropdown_selected,
            'model_dropdown_selected': model_dropdown_selected,
            'dataset_dropdown_selected': dataset_dropdown_selected,
            'prediction': prediction,
            'prediction_filter_selected': prediction_filter_selected,
            'prediction_data_type_selected': prediction_data_type_selected,
            'dataloader_obj': dataloader_obj,
        }
        return render(request, 'inference.html', context)

def download_prediction(request):
    """ Function: download_prediction
     * download prediction
    """
    
    # --- get selected project and model ---
    selected_project_name = request.session.get('inference_view_selected_project', None)
    selected_model_name = request.session.get('inference_view_selected_model', None)
    
    selected_project = Project.objects.get(name=selected_project_name)
    selected_model = MlModel.objects.get(name=selected_model_name, project=selected_project)
    
    # --- get prediction ---
    prediction_data_type_selected = request.session.get('prediction_data_type', 'Test')
    prediction_csv = Path(selected_model.model_dir, f'{prediction_data_type_selected.lower()}_prediction.csv')
    
    return FileResponse(open(prediction_csv, "rb"), as_attachment=True, filename=prediction_csv.name)
    
