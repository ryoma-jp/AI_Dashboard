import os
import logging
import cv2
import time
import requests
import tarfile
import numpy as np

from pathlib import Path

from machine_learning.lib.predictor.predictor_keras import Predictor
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
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # --- Prepare model for inference ---
            if (streaming_model in pretrained_model_list):
                # --- load model ---
                pretrained_model = Predictor(model_name=streaming_model)
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
                    dsize = pretrained_model.input_shape[0:2]
                    overlay_for_inference = cv2.resize(overlay, dsize=dsize, interpolation=cv2.INTER_AREA)
                    overlay_for_inference = cv2.cvtColor(overlay_for_inference, cv2.COLOR_BGR2RGB)
                    
                    overlay_for_inference = np.asarray(overlay_for_inference, dtype='float32')
                    overlay_for_inference = overlay_for_inference[np.newaxis]
                    
                    # --- Infrerence ---
                    preds = pretrained_model.predict(overlay_for_inference)
                    #print(preds)
                    #print(preds.shape)
                    #print(np.argmax(preds[0]))
                    """SampleCode
                    import tensorflow as tf
                    import numpy as np
                    
                    from PIL import Image
                    from tensorflow import keras
                    
                    img = Image.open('00000000.png')
                    img = img.resize([224, 224])
                    print(img.size)
                    
                    img = np.asarray(img, dtype='float32')
                    img = img[np.newaxis]
                    print(img.shape)
                    tf_data = tf.image.convert_image_dtype(img, dtype=tf.float32) * (2.0 / 255.0) - 1.0
                    pretrained_model = keras.applications.ResNet50()
                    
                    preds = pretrained_model.predict(tf_data)
                    print(preds)
                    print(preds.shape)
                    print(np.argmax(preds[0]))
                    
                    """
                
                # --- Put Text(FPS) ---
                time_end = time.time()
                processing_rate = 1.0 / (time_end - time_start)
                fps_text = f'fps : {processing_rate:.02f}'
                cv2.rectangle(overlay, (0, 0), (100, 20), (0, 0, 0), -1)
                cv2.putText(overlay, fps_text, fps_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(250,225,0))
                
                # --- Put Text(Model) ---
                cv2.rectangle(overlay, (0, 20), (200, 40), (0, 0, 0), -1)
                cv2.putText(overlay, model_name_text, model_name_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(100,250,0))
                
                # --- Put Text(Class) ---
                if (streaming_model in pretrained_model_list):
                    class_text = f'id={np.argmax(preds[0])}'
                    cv2.rectangle(overlay, (0, 40), (200, 60), (0, 0, 0), -1)
                    cv2.putText(overlay, class_text, class_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,250,225))
                
                # --- Alpha and Concat---
                frame = cv2.hconcat([frame, cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)])
                
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
