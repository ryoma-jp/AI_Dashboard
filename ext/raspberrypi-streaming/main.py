
import os
import cv2
import numpy as np

from lib.inference_tflite import object_detection

class StreamingFilter:
    '''Filter of mjpg-streamer for Streaming 
    '''

    def __init__(self, tflite_file=None, class_label=None):
        self.debug = 0
        self.tflite_file = tflite_file

        with open(class_label, 'r') as f:
            self.class_label = f.read().splitlines()

    def path_through(self, img):
        '''
            :param img: A numpy array representing the input image
            :returns: A numpy array to send to the mjpg-streamer output plugin
        '''
        return img

    def object_detection(self, img):
        '''
            :param img: A numpy array representing the input image
            :returns: A numpy array to send to the mjpg-streamer output plugin
        '''
        ret_img, time = object_detection(self.tflite_file, self.class_label, img)

        if (self.debug == 0):
            print(time)
            self.debug += 1


        return img
        
def init_filter():
    '''
        This function is called after the filter module is imported. It MUST
        return a callable object (such as a function or bound method). 
    '''
    task = os.environ['STREAMING_TASK']
    f = StreamingFilter(tflite_file=os.environ['TFLITE_FILE'], class_label=os.environ['CLASS_LABEL'])

    if (task == 'object_detection'):
        filter_fn = f.object_detection
    else:
        filter_fn = f.path_through

    return filter_fn

