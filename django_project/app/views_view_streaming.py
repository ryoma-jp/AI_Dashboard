import os
import logging
import cv2
import time

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
        cap = cv2.VideoCapture(0)
        if (not cap.isOpened()):
            # --- Retry ---
            cap.release()
            time.sleep(3)
            cap = cv2.VideoCapture(0)
        
        if (cap.isOpened()):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            
            while True:
                # --- Get Frame ---
                time_start = time.time()
                ret, frame = cap.read()
                overlay = frame.copy()
                
                # --- Put Text ---
                time_end = time.time()
                processing_rate = 1.0 / (time_end - time_start)
                text = f'fps : {processing_rate:.02f}'
                org = (5, 15)
                cv2.rectangle(overlay, (0, 0), (100, 20), (0, 0, 0), -1)
                cv2.putText(overlay, text, org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(250,225,0))
                
                # --- Alpha and Concat---
                alpha = 0.7
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
