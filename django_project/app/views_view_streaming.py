import sys
import os
import logging
import cv2
import time
import json
import requests
import tarfile
import numpy as np
import validators

from pathlib import Path
from PIL import Image
from cap_from_youtube import cap_from_youtube

from machine_learning.lib.predictor.predictor_keras import PredictorMlModel, PredictorResNet50, PredictorCenterNetHourGlass104
from machine_learning.lib.utils.utils import download_file, safe_extract_tar, zip_extract

from app.models import Project, MlModel

from django.shortcuts import render, redirect
from django.http.response import StreamingHttpResponse

from views_common import SidebarActiveStatus, get_version, get_jupyter_nb_url

# Create your views here.

def _get_model_for_inference(request, streaming_project_name, streaming_model_name):
    """Get Model for Inference
    
    This function is internal in the streaming view, gets the model for inference.
    
    Args:
        streaming_project_name (string): project name in Streaming view
        streaming_model_name (string): model name in Streaming view
        
    Returns:
        model name and model object
            - streaming_model_name
            - pretrained_model
    """
    
    category_names = None
    if (streaming_project_name == 'Sample'):
        if (streaming_model_name == 'ResNet50'):
            # --- load model ---
            pretrained_model = PredictorResNet50()
            logging.info('-------------------------------------')
            logging.info(pretrained_model.pretrained_model)
            logging.info('-------------------------------------')
        elif (streaming_model_name == 'CenterNetHourGlass104'):
            # --- Parameters to draw detection result ---
            if (not Path('/tmp/annotations/instances_val2017.json').exists()):
                url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
                download_file(url, save_dir='/tmp')
                zip_extract('/tmp/annotations_trainval2017.zip', '/tmp')
            with open('/tmp/annotations/instances_val2017.json', 'r') as f:
                instances_val2017 = json.load(f)
                category_names = {data_['id']: data_['name'] for data_ in instances_val2017['categories']}
            
            # --- Create object of Pre-trained model ---
            pretrained_model = PredictorCenterNetHourGlass104()
            
            # --- T.B.D ---
            #   * CenterNetHourGlass104 has no attribute 'summary'
            #       - AttributeError: '_UserObject' object has no attribute 'summary'
            #logging.info('-------------------------------------')
            #logging.info(f'model_summary')
            #pretrained_model.pretrained_model.summary(print_fn=logging.info)
            #logging.info('-------------------------------------')
            
        else:
            pretrained_model = None
            logging.info('-------------------------------------')
            logging.info(f'Unknown streaming model: streaming_model_name={streaming_model_name}')
            logging.info('-------------------------------------')
    else:
        streaming_project = Project.objects.get(name=streaming_project_name)
        if (streaming_model_name in [f.name for f in MlModel.objects.filter(project=streaming_project)]):
            streaming_model = MlModel.objects.get(name=streaming_model_name, project=streaming_project)
            
            ## --- Create object of Pre-trained model ---
            #get_feature_map = True
            #if (request.session.get('show_features_enable_selected', 'False') == 'False'):
            #    get_feature_map = False
            #feature_map_calc_range = request.session.get('show_features_calc_range_selected', 'Model-wise')
            #pretrained_model = PredictorMlModel(streaming_model, get_feature_map=get_feature_map, feature_map_calc_range=feature_map_calc_range)
            #
            #if (len(pretrained_model.category_list) > 0):
            #    category_names = {key: item for key, item in zip(list(range(0, len(pretrained_model.category_list))), pretrained_model.category_list)}
            #
            #logging.info('-------------------------------------')
            #logging.info(f'model_summary')
            #pretrained_model.pretrained_model.summary(print_fn=logging.info)
            #logging.info(f'category_names')
            #logging.info(category_names)
            #logging.info('-------------------------------------')

            # --- import AI Model SDK ---
            sys.path.append(streaming_model.ai_model_sdk.ai_model_sdk_dir)
            from ai_model_sdk import AI_Model_SDK
            logging.info(AI_Model_SDK.__version__)

            config_path = Path(streaming_model.model_dir, 'config.json')
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            model_params = {
                'model_path': streaming_model.model_dir,
            }
            dataset_path = config_data['dataset']['dataset_dir']['value']
            dataset = Path(dataset_path, 'dataset.pkl')
            pretrained_model = AI_Model_SDK(dataset, model_params)

            get_feature_map = True
            if (request.session.get('show_features_enable_selected', 'False') == 'False'):
                get_feature_map = False
            feature_map_calc_range = request.session.get('show_features_calc_range_selected', 'Model-wise')

            trained_model = Path(streaming_model.model_dir, 'models')
            pretrained_model.load_model(trained_model, get_feature_map=get_feature_map, feature_map_calc_range=feature_map_calc_range)
            
        else:
            streaming_model_name = 'None'
            pretrained_model = None

    return streaming_model_name, pretrained_model, category_names

def _create_frame(frame, overlay, 
                  pretrained_model, 
                  fps_text, fps_org, model_name_text, model_name_org, class_org, alpha,
                  height, width, 
                  category_names):
    """Create Frame
    
    This function is internal in the streaming view, creates the frame to draw in the streaming view.
    
    Args:
        frame (numpy.ndarray): original camera frame
        overlay (numpy.ndarray): copy of camera frame
        pretrained_model (MlModel): pre-trained model object
        fps_text (string): framerate
        fps_org (list): position of ``fps_text``
        model_name_text (string): model name
        model_name_org (list): position of ``model_name_text``
        class_org (list): position of class text
        alpha (float): blend alpha for drawing the prediction result
        height (int): image height
        width (int): image width
        category_names (list): category name list
    """
    if (pretrained_model is not None):
        if (pretrained_model.task == 'classification'):
            prediction_area = np.zeros([height, 320, 3], np.uint8)
            cv2.putText(prediction_area, fps_text, fps_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(250,225,0))
            cv2.putText(prediction_area, model_name_text, model_name_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(100,225,0))
            
            cv2.putText(prediction_area, 'class_name:', class_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,250,225))
            for i in range(len(pretrained_model.decoded_preds["class_name"])):
                class_text_ = f'  * {pretrained_model.decoded_preds["class_name"][i]}'
                class_org_ = (class_org[0], class_org[1] + (i+1)*20)
                cv2.putText(prediction_area, class_text_, class_org_, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,250,225))
            
            frame = cv2.hconcat([frame, prediction_area])
        elif ('object_detection' in pretrained_model.task):
            information_area = np.zeros([height, 320, 3], np.uint8)
            cv2.putText(information_area, fps_text, fps_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(250, 225, 0))
            cv2.putText(information_area, model_name_text, model_name_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(100, 225, 0))
            
            if (len(pretrained_model.decoded_preds['detection_boxes']) > 0):
                boxes = np.asarray(pretrained_model.decoded_preds['detection_boxes'] * [height, width, height, width], dtype=int)
                for box_, class_, score_ in zip(boxes, pretrained_model.decoded_preds['detection_classes'], pretrained_model.decoded_preds['detection_scores']):
                    cv2.rectangle(overlay, [box_[1], box_[0]], [box_[3], box_[2]], color=[255, 0, 0])
                    if ((pretrained_model.category_names is not None) and (len(pretrained_model.category_names) > 0)):
                        cv2.putText(overlay,
                                    pretrained_model.category_names[class_],
                                    [box_[1], box_[0]-8],
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4, thickness=2, color=(255, 0, 0))
                    else:
                        cv2.putText(overlay,
                                    f'class{class_}',
                                    [box_[1], box_[0]-8],
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4, thickness=2, color=(255, 0, 0))
                
            frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
            frame = cv2.hconcat([frame, information_area])
        else:
            cv2.rectangle(overlay, (0, 0), (100, 20), (0, 0, 0), -1)
            cv2.putText(overlay, fps_text, fps_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(250,225,0))
            cv2.rectangle(overlay, (0, 20), (200, 40), (0, 0, 0), -1)
            cv2.putText(overlay, model_name_text, model_name_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(100,250,0))
            frame = cv2.hconcat([frame, cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)])
    
        if (pretrained_model.get_feature_map):
            feature_map = pretrained_model.create_feature_map()
            
            width_sub = frame.shape[1] - feature_map.shape[1]
            if (width_sub > 0):
                feature_map = cv2.hconcat([feature_map, np.full([feature_map.shape[0], abs(width_sub), 3], 255, dtype=np.uint8)])
            elif (width_sub < 0):
                frame = cv2.hconcat([frame, np.zeros([frame.shape[0], abs(width_sub), 3], np.uint8)])
            
            frame = cv2.vconcat([frame, feature_map])
    
    else:
        cv2.rectangle(overlay, (0, 0), (100, 20), (0, 0, 0), -1)
        cv2.putText(overlay, fps_text, fps_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(250,225,0))
        cv2.rectangle(overlay, (0, 20), (200, 40), (0, 0, 0), -1)
        cv2.putText(overlay, model_name_text, model_name_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(100,250,0))
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    
    return frame
    
def view_streaming(request):
    """ Function: view_streaming
     * view_streaming top
    """
    
    def _check_url(ip_addr, port):
        """_check_url
        [T.B.D] return True if IP address and PORT is valid, else return False
        """
        if ((ip_addr[0] == '192') and
            (ip_addr[1] == '168')):
            return True
        else:
            return False
    
    if (request.method == 'POST'):
        logging.info('-------------------------------------')
        logging.info(request.method)
        logging.info(request.POST)
        logging.info('-------------------------------------')
        
        if ('streaming_view_interface_dropdown' in request.POST):
            request.session['streaming_interface'] = request.POST.getlist('streaming_view_interface_dropdown')[0]
        elif ('streaming_view_project_dropdown' in request.POST):
            request.session['streaming_selected_project'] = request.POST.getlist('streaming_view_project_dropdown')[0]
        elif ('streaming_view_model_dropdown' in request.POST):
            request.session['streaming_selected_model'] = request.POST.getlist('streaming_view_model_dropdown')[0]
        elif ('view_streaming_usbcam_apply' in request.POST):
            tmp_val = request.POST.getlist('streaming_show_features_enable_selected_submit')[0]
            if (tmp_val != ''):
                request.session['show_features_enable_selected'] = tmp_val
            
            tmp_val = request.POST.getlist('streaming_show_features_calc_range_selected_submit')[0]
            if (tmp_val != ''):
                request.session['show_features_calc_range_selected'] = tmp_val
            
        elif ('view_streaming_youtube_url_apply' in request.POST):
            request.session['streaming_youtube_url'] = request.POST.getlist('view_streaming_youtube_url')[0]
            
            tmp_val = request.POST.getlist('streaming_show_features_enable_selected_submit')[0]
            if (tmp_val != ''):
                request.session['show_features_enable_selected'] = tmp_val
            
            tmp_val = request.POST.getlist('streaming_show_features_calc_range_selected_submit')[0]
            if (tmp_val != ''):
                request.session['show_features_calc_range_selected'] = tmp_val
            
        elif ('view_streaming_ipaddr_apply' in request.POST):
            ip_addr = [
                request.POST['ip_0'],
                request.POST['ip_1'],
                request.POST['ip_2'],
                request.POST['ip_3'],
            ]
            request.session['ip_addr'] = ip_addr
            request.session['port'] = request.POST['port']
    
    project = Project.objects.all().order_by('-id').reverse()
    model = MlModel.objects.all().order_by('-id').reverse()
    
    streaming_interface = request.session.get('streaming_interface', None)
    streaming_selected_project = request.session.get('streaming_selected_project', 'Sample')
    streaming_selected_model = request.session.get('streaming_selected_model', None)
    show_features_enable_selected = request.session.get('show_features_enable_selected', 'False')
    show_features_calc_range_selected = request.session.get('show_features_calc_range_selected', 'Model-wise')

    streaming_model_name, pretrained_model, category_names = None, None, None
    if (streaming_selected_project in [proj.name for proj in Project.objects.all()]):
        streaming_model_name, pretrained_model, category_names = _get_model_for_inference(request, streaming_selected_project, streaming_selected_model)
    if (pretrained_model is not None):
        feature_name_list = pretrained_model.feature_name_list
    else:
        feature_name_list = []

    get_feature_map = True
    if (show_features_enable_selected == 'False'):
        get_feature_map = False
    if (get_feature_map and (not pretrained_model.get_feature_map)):
        show_features_supported_model = 'False'
    else:
        show_features_supported_model = 'True'
    
    if (streaming_selected_project in [proj.name for proj in Project.objects.all()]):
        selected_project = Project.objects.get(name=streaming_selected_project)
        pretrained_model_list = [model.name for model in MlModel.objects.filter(project=selected_project)]
    else:
        streaming_selected_project = 'Sample'
        pretrained_model_list = [
            'ResNet50',
            'CenterNetHourGlass104',
        ]
    
    request.session['pretrained_model_list'] = pretrained_model_list
    
    if (streaming_selected_project in [proj.name for proj in Project.objects.all()]):
        selected_project = Project.objects.get(name=streaming_selected_project)
    
        if (streaming_selected_model not in [model.name for model in MlModel.objects.filter(project=selected_project)]):
            streaming_selected_model = None
    else:
        selected_project = None
        if (streaming_selected_model not in pretrained_model_list):
            streaming_selected_model = 'None'
    
    ip_addr = request.session.get('ip_addr', ['0', '0', '0', '0'])
    port = request.session.get('port', '0')
    valid_url = _check_url(ip_addr, port)
    
    default_url = 'https://www.youtube.com/watch?v=Ii8u5eywxgI'
    youtube_url = request.session.get('streaming_youtube_url', default_url)
    if (not validators.url(youtube_url)):
        # --- if got the invalid url, set default_url ---
        request.session['streaming_youtube_url'] = default_url
        youtube_url = default_url
    else:
        try:
            cap_from_youtube(url, resolution)
        except:
            request.session['streaming_youtube_url'] = default_url
            youtube_url = default_url
    
    sidebar_status = SidebarActiveStatus()
    sidebar_status.view_streaming = 'active'
        
    context = {
        'sidebar_status': sidebar_status,
        'text': get_version(),
        'jupyter_nb_url': get_jupyter_nb_url(),
        'project': project,
        'model': model,
        'streaming_interface': streaming_interface,
        'streaming_view_selected_project': streaming_selected_project,
        'streaming_view_selected_project_obj': selected_project,
        'streaming_view_selected_model': streaming_selected_model,
        'pretrained_model_list': pretrained_model_list,
        'ip_addr': ip_addr,
        'port': port,
        'valid_url': valid_url,
        'youtube_url': youtube_url,
        'show_features_enable_selected': show_features_enable_selected,
        'show_features_supported_model': show_features_supported_model,
        'show_features_calc_range_selected': show_features_calc_range_selected,
        'show_features_name_list': feature_name_list,
    }
    return render(request, 'view_streaming.html', context)


def usb_cam(request):
    """ Function: usb_cam
     * USB Camera streaming
    """
    
    def gen():
        """Generator
        
        This function is generator of byte frame for StreamingHttpResponse
        """
        streaming_project_name = request.session.get('streaming_selected_project', 'Sample')
        streaming_model_name = request.session.get('streaming_selected_model', 'None')
        pretrained_model_list = request.session.get('pretrained_model_list', [''])
        
        cap = cv2.VideoCapture(0)
        if (not cap.isOpened()):
            # --- Retry ---
            cap.release()
            time.sleep(3)    # weight for complete ``cap.release`` (3sec is tentative)
            cap = cv2.VideoCapture(0)
        
        if (cap.isOpened()):
            width = 320
            height = 240
            fps = 30
            
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            
            # --- Prepare model for inference ---
            streaming_model_name, pretrained_model, category_names = _get_model_for_inference(request, streaming_project_name, streaming_model_name)
            
            # --- Set fixed parameters ---
            fps_org = (5, 15)
            model_name_org = (5, 35)
            model_name_text = f'Model: {streaming_model_name}'
            alpha = 0.8
            
            class_org = None
            if ((pretrained_model is not None) and (pretrained_model.task == 'classification')):
                # --- result area to draw the predicted class---
                class_org = (5, 55)
            
            while True:
                # --- Get Frame ---
                time_start = time.time()
                ret, frame = cap.read()
                overlay = frame.copy()
                
                if (streaming_model_name in pretrained_model_list):
                    # --- Convert format ---
                    overlay_for_inference = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                    #overlay_for_inference = overlay_for_inference.resize(pretrained_model.input_shape[0:2])
                    #overlay_for_inference = np.expand_dims(np.asarray(overlay_for_inference), axis=0)
                    
                    # --- Infrerence ---
                    pred = pretrained_model.predict(overlay_for_inference)
                    pretrained_model.decode_prediction(pred)
                            
                # --- Put Text ---
                time_end = time.time()
                processing_rate = 1.0 / (time_end - time_start)
                fps_text = f'fps : {processing_rate:.02f}'
                
                frame = _create_frame(frame, overlay,
                      pretrained_model,
                      fps_text, fps_org, model_name_text, model_name_org, class_org, alpha,
                      height, width, 
                      category_names)
                
                # --- Encode and Return byte frame ---
                image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n\r\n')
        else:
            logging.info('-------------------------------------')
            logging.info('cap.isOpened() is False')
            logging.info('-------------------------------------')

    logging.info('-------------------------------------')
    logging.info(request.method)
    logging.info(request.POST)
    logging.info('-------------------------------------')
    
    return StreamingHttpResponse(gen(),
               content_type='multipart/x-mixed-replace; boundary=frame')

def youtube(request):
    """ Function: youtube
     * Youtube archive
    """
    
    def gen():
        """Generator
        
        This function is generator of byte frame for StreamingHttpResponse
        """
        streaming_project_name = request.session.get('streaming_selected_project', 'Sample')
        streaming_model_name = request.session.get('streaming_selected_model', 'None')
        pretrained_model_list = request.session.get('pretrained_model_list', [''])
        
        resolution = '240p'    # T.B.D
        default_url = 'https://www.youtube.com/watch?v=Ii8u5eywxgI'
        url = request.session.get('streaming_youtube_url', default_url)
        
        cap = cap_from_youtube(url, resolution)
        
        width = 427
        height = 240
        fps = 30
        frame_duration = 1 / fps
        
        # --- Prepare model for inference ---
        streaming_model_name, pretrained_model, category_names = _get_model_for_inference(request, streaming_project_name, streaming_model_name)

        # --- Set fixed parameters ---
        fps_org = (5, 15)
        model_name_org = (5, 35)
        model_name_text = f'Model: {streaming_model_name}'
        alpha = 0.8
        
        class_org = None
        if ((pretrained_model is not None) and (pretrained_model.task == 'classification')):
            # --- result area to draw the predicted class---
            class_org = (5, 55)
        
        while True:
            # --- Get Frame ---
            time_start = time.time()
            ret, frame = cap.read()
            if (not ret):
                cap = cap_from_youtube(url, resolution)
                continue
            
            overlay = frame.copy()
            
            #logging.info('-------------------------------------')
            #logging.info('[DEBUG]')
            #logging.info(f'  * streaming_model_name = {streaming_model_name}')
            #logging.info(f'  * pretrained_model_list = {pretrained_model_list}')
            #logging.info('-------------------------------------')
            if (streaming_model_name in pretrained_model_list):
                # --- Convert format ---
                    overlay_for_inference = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                    #overlay_for_inference = overlay_for_inference.resize(pretrained_model.input_shape[0:2])
                    #overlay_for_inference = np.expand_dims(np.asarray(overlay_for_inference), axis=0)
                    
                    # --- Infrerence ---
                    pred = pretrained_model.predict(overlay_for_inference)
                    pretrained_model.decode_prediction(pred)
                
            # --- Put Text ---
            time_end = time.time()
            processing_rate = 1.0 / (time_end - time_start)
            fps_text = f'fps : {processing_rate:.02f}'
            
            sleep_time = frame_duration - (time_end - time_start)
            if (sleep_time > 0):
                time.sleep(sleep_time)
            
            frame = _create_frame(frame, overlay,
                                  pretrained_model,
                                  fps_text, fps_org, model_name_text, model_name_org, class_org, alpha,
                                  height, width, 
                                  category_names)

            # --- Encode and Return byte frame ---
            image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n\r\n')

    logging.info('-------------------------------------')
    logging.info(request.method)
    logging.info(request.POST)
    logging.info('-------------------------------------')
    
    return StreamingHttpResponse(gen(),
               content_type='multipart/x-mixed-replace; boundary=frame')
