"""USB Camera test

This file describe about the USB camera test
"""

import cv2
import time
import argparse

def ArgParser():
    """Argument Parser
    
    This function parses arguments of this tool.
    """
    parser = argparse.ArgumentParser(description='Test code of USB camera',
                formatter_class=argparse.RawTextHelpFormatter)

    args = parser.parse_args()

    return args

def main():
    """Main
    
    This function is main routine.
    """
    
    width = 320
    height = 240
    fps = 30
    dev_id = 0
    
    cap = cv2.VideoCapture(dev_id)
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    while True:
        time_start = time.time()
        ret, frame = cap.read()
        overlay = frame.copy()
        
        time_end = time.time()
        processing_rate = 1.0 / (time_end - time_start)
        text = f'fps : {processing_rate:.02f}'
        org = (5, 15)
        cv2.rectangle(overlay, (0, 0), (100, 20), (0, 0, 0), -1)
        cv2.putText(overlay, text, org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(250,225,0))
        
        alpha = 0.7
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
    