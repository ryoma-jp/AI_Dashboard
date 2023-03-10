import os
import logging
import cv2
import time
import requests
import tarfile
import numpy as np

from pathlib import Path
from PIL import Image

from machine_learning.lib.predictor.predictor_keras import PredictorResNet50
from machine_learning.lib.utils.utils import download_file, safe_extract_tar

from django.shortcuts import render, redirect
from django.http.response import StreamingHttpResponse

from views_common import SidebarActiveStatus, get_version, get_jupyter_nb_url

# Create your views here.

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
        
        if ('streaming_interface_dropdown' in request.POST):
            request.session['streaming_interface'] = request.POST.getlist('streaming_interface_dropdown')[0]
        elif ('streaming_model_dropdown' in request.POST):
            request.session['streaming_model'] = request.POST.getlist('streaming_model_dropdown')[0]
        elif ('view_streaming_apply' in request.POST):
            ip_addr = [
                request.POST['ip_0'],
                request.POST['ip_1'],
                request.POST['ip_2'],
                request.POST['ip_3'],
            ]
            request.session['ip_addr'] = ip_addr
            request.session['port'] = request.POST['port']
    
    streaming_interface = request.session.get('streaming_interface', None)
    streaming_model = request.session.get('streaming_model', None)
    pretrained_model_list = [
        'ResNet50',
    ]
    request.session['pretrained_model_list'] = pretrained_model_list
    
    ip_addr = request.session.get('ip_addr', ['0', '0', '0', '0'])
    port = request.session.get('port', '0')
    valid_url = _check_url(ip_addr, port)
    
    sidebar_status = SidebarActiveStatus()
    sidebar_status.view_streaming = 'active'
        
    context = {
        'sidebar_status': sidebar_status,
        'text': get_version(),
        'jupyter_nb_url': get_jupyter_nb_url(),
        'streaming_interface': streaming_interface,
        'streaming_model': streaming_model,
        'pretrained_model_list': pretrained_model_list,
        'ip_addr': ip_addr,
        'port': port,
        'valid_url': valid_url,
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
        streaming_model = request.session.get('streaming_model', 'None')
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
            if (streaming_model == 'ResNet50'):
                # --- load model ---
                pretrained_model = PredictorResNet50()
                logging.info('-------------------------------------')
                logging.info(pretrained_model.pretrained_model)
                logging.info('-------------------------------------')
            else:
                pretrained_model = None
                logging.info('-------------------------------------')
                logging.info(f'Unknown streaming model: streaming_model={streaming_model}')
                logging.info('-------------------------------------')
                
            # --- Set fixed parameters ---
            fps_org = (5, 15)
            model_name_org = (5, 35)
            if (streaming_model in pretrained_model_list):
                model_name_text = f'Model: {streaming_model}'
            else:
                model_name_text = f'Model: None'
            class_org = (5, 55)
            alpha = 0.8
            
            while True:
                # --- Get Frame ---
                time_start = time.time()
                ret, frame = cap.read()
                overlay = frame.copy()
                
                if (streaming_model in pretrained_model_list):
                    # --- Convert format ---
                    overlay_for_inference = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                    overlay_for_inference = overlay_for_inference.resize(pretrained_model.input_shape[0:2])
                    overlay_for_inference = np.expand_dims(np.asarray(overlay_for_inference), axis=0)
                    
                    # --- Infrerence ---
                    preds = pretrained_model.predict(overlay_for_inference)
                    pretrained_model.decode_predictions(preds)
                
                # --- Put Text ---
                time_end = time.time()
                processing_rate = 1.0 / (time_end - time_start)
                fps_text = f'fps : {processing_rate:.02f}'
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
                    else:
                        cv2.rectangle(overlay, (0, 0), (100, 20), (0, 0, 0), -1)
                        cv2.putText(overlay, fps_text, fps_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(250,225,0))
                        cv2.rectangle(overlay, (0, 20), (200, 40), (0, 0, 0), -1)
                        cv2.putText(overlay, model_name_text, model_name_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(100,250,0))
                        frame = cv2.hconcat([frame, cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)])
                else:
                    cv2.rectangle(overlay, (0, 0), (100, 20), (0, 0, 0), -1)
                    cv2.putText(overlay, fps_text, fps_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(250,225,0))
                    cv2.rectangle(overlay, (0, 20), (200, 40), (0, 0, 0), -1)
                    cv2.putText(overlay, model_name_text, model_name_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(100,250,0))
                    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

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
