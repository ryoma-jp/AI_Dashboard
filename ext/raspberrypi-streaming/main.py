
import os
import cv2
import numpy as np

class StreamingFilter:
    '''Filter of mjpg-streamer for Streaming 
    '''

    def __init__(self, tflite_file=None):
        self.debug = 0
        self.tflite_file = tflite_file

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

        return img
        
def init_filter():
    '''
        This function is called after the filter module is imported. It MUST
        return a callable object (such as a function or bound method). 
    '''
    f = StreamingFilter(tflite_file=os.environ['TFLITE_FILE'])
    return f.object_detection

