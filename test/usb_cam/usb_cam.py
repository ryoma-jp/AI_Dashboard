"""USB Camera test

This file describe about the USB camera test
"""

import cv2
import time
import argparse
import numpy as np

from PIL import Image
from machine_learning.lib.predictor.predictor_keras import PredictorResNet50

def ArgParser(model_list):
    """Argument Parser
    
    This function parses arguments of this tool.
    
    Args:
        model_list (list of string): string list for model selection
    """
    parser = argparse.ArgumentParser(description='Test code of USB camera',
                formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        choices=model_list,
        default='None',
        required=False,
        help='Choose the model for prediction \n'
             '  - None: show camera image only\n'
             '  - ResNet50: image classification using ResNet50')
    
    args = parser.parse_args()

    return args

def main():
    """Main
    
    This function is main routine.
    """
    model_list = ['None', 'ResNet50']
    args = ArgParser(model_list)
    print(f'args.model : {args.model}')
    
    width = 320
    height = 240
    fps = 30
    dev_id = 0
    
    cap = cv2.VideoCapture(dev_id)
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    fps_org = (5, 15)
    model_name_org = (5, 35)
    model_name_text = f'Model: {args.model}'
    class_org = (5, 55)
    alpha = 0.8
    
    if (args.model == 'ResNet50'):
        pretrained_model = PredictorResNet50()
    else:
        pretrained_model = None
    
    while True:
        time_start = time.time()
        ret, frame = cap.read()
        overlay = frame.copy()
        
        if (pretrained_model is not None):
            if (pretrained_model.task == 'classification'):
                # --- Pre-processing ---
                overlay_for_inference = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                overlay_for_inference = overlay_for_inference.resize(pretrained_model.input_shape[0:2])
                overlay_for_inference = np.expand_dims(np.asarray(overlay_for_inference), axis=0)
                
                # --- Infrerence ---
                preds = pretrained_model.predict(overlay_for_inference)
                pretrained_model.decode_predictions(preds)
        
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
                frame = cv2.hconcat([frame, cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)])
        else:
            cv2.rectangle(overlay, (0, 0), (100, 20), (0, 0, 0), -1)
            cv2.putText(overlay, fps_text, fps_org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(250,225,0))
            frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
            
        if not ret:
            continue
        cv2.imshow('usb cam test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    main()
    