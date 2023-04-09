"""Predictor for Keras

This file describe about the prediction process using Keras
"""

import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from pathlib import Path
from PIL import Image
from tensorflow import keras

from machine_learning.lib.utils.utils import download_file, safe_extract_tar

class Predictor():
    """Predictor
    
    This class specifies the process of loading model and predicting.
    """
    
    def __init__(self, task):
        """Constructor
        
        This function is the construction of predictor.
        
        Args:
            task (string): task of model
                - 'classification'
        """
        if (task == 'classification'):
            self.decoded_preds = {
                'class_id': None,
                'class_name': None,
                'score': None,
            }
        elif (task == 'object_detection'):
            self.decoded_preds = {
                'num_detections': None,
                'detection_boxes': None,
                'detection_classes': None,
                'detection_scores': None,
            }
        else:
            self.decoded_preds = {}
    
    def predict(self, x):
        """Predict
        
        This function predicts ``x`` using ``self.pretrained_model``
        
        Args:
            x (np.array): input data
                            - image: shape is [[N]+``self.input_shape``], channel is [R, G, B]
        
        Return:
            prediction as dict object
        """
        
        prediction = {}
        
        return prediction
    
class PredictorMlModel(Predictor):
    """Predictor
    
    This class specifies the process of loading model and predicting.
    The pre-trainded model is trained by AI Dashboard.
    
    """
    
    def __init__(self, mlmodel, input_shape, task):
        """Constructor
        
        This function is the construction of predictor.
        
        Args:
            mlmodel (object): MlModel class
            input_shape (list): shape of input tensor
        """
        
        # --- initialize ---
        if (task in ['img_clf', 'reg_clf']):
            self.task = 'classification'
        else:
            self.task = 'object_detection'
        super().__init__(self.task)
        self.input_shape = input_shape
        
        # --- load model ---
        trained_model_path = Path(mlmodel.model_dir, 'models', 'hdf5', 'model.h5')
        self.pretrained_model = keras.models.load_model(trained_model_path)
        
        # --- load config and set parameters ---
        config_path = Path(mlmodel.model_dir, 'config.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            self.input_shape = config_data['inference_parameter']['preprocessing']['input_shape']['value']
            self.norm_coef_a = config_data['inference_parameter']['preprocessing']['norm_coef_a']['value']
            self.norm_coef_b = config_data['inference_parameter']['preprocessing']['norm_coef_b']['value']
    
    def preprocess_input(self, x):
        """Preprocess input data
        
        This function pre-processes(cropping, scaling, etc) the input data ``x``.
        
        Args:
            x (np.array): input data
        
        Returns:
            pre-processed data as np.array
        """
        
        y = (x.astype(float) - self.norm_coef_a) / self.norm_coef_b
        
        return y
    
    def predict(self, x):
        """Predict
        
        This function predicts ``x`` using ``self.pretrained_model``
        
        Args:
            x (np.array): input data
                            - image: shape is [[N]+``self.input_shape``], channel is [R, G, B]
        
        Returns:
            prediction as np.array
        """
        
        prediction = self.pretrained_model.predict(self.preprocess_input(x))
        
        return prediction
    
    def decode_predictions(self, preds):
        """Decode Predictions
        
        This function converts predictions to dict format
        
        Args:
            preds (np.array): predictions
        
        Return:
            
        """
        
        top5_score = np.sort(preds[0])[::-1][0:5]
        top5_class_id = np.argsort(preds[0])[::-1][0:5]
        
        self.decoded_preds['class_id'] = top5_class_id
        self.decoded_preds['class_name'] = [f'class{i}' for i in top5_class_id]
        self.decoded_preds['score'] = top5_score

class PredictorResNet50(Predictor):
    """Predictor
    
    This class specifies the process of loading model and predicting.
    """
    
    def __init__(self):
        """Constructor
        
        This function is the construction of predictor.
        
        """
        super().__init__('classification')
        
        self.task = 'classification'
        self.pretrained_model = keras.applications.ResNet50()
        self.input_shape = [224, 224, 3]
    
    def preprocess_input(self, x):
        """Preprocess input data
        
        This function pre-processes(cropping, scaling, etc) the input data ``x``.
        
        Args:
            x (np.array): input data
        
        Returns:
            pre-processed data as np.array
        """
        
        y = x
        
        return y
        
    def predict(self, x):
        """Predict
        
        This function predicts ``x`` using ``self.pretrained_model``
        
        Args:
            x (np.array): input data
                            - image: shape is [[N]+``self.input_shape``], channel is [R, G, B]
        
        Return:
            prediction as np.array
        """
        
        preprocessing_input_ = keras.applications.resnet50.preprocess_input
        prediction = self.pretrained_model.predict(preprocessing_input_(x))
        
        return prediction
    
    def decode_predictions(self, preds):
        """Decode Predictions
        
        This function converts predictions to dict format
        
        Args:
            preds (np.array): predictions
        
        Return:
            
        """
        decode_predictions = keras.applications.resnet50.decode_predictions
        decoded_preds_ = decode_predictions(preds, top=5)[0]
        
        self.decoded_preds['class_id'] = [decoded_preds_[i][0] for i in range(len(decoded_preds_))]
        self.decoded_preds['class_name'] = [decoded_preds_[i][1] for i in range(len(decoded_preds_))]
        self.decoded_preds['score'] = [decoded_preds_[i][2] for i in range(len(decoded_preds_))]

class PredictorCenterNetHourGlass104(Predictor):
    """Predictor
    
    This class specifies the process of loading model and predicting.
    """
    
    def __init__(self):
        """Constructor
        
        This function is the construction of predictor.
        
        """
        super().__init__('object_detection')
        
        self.task = 'object_detection'
        
        url = 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1'
        self.pretrained_model = hub.load(url)
        self.input_shape = [512, 512, 3]
        
    def preprocess_input(self, x):
        """Preprocess input data
        
        This function pre-processes(cropping, scaling, etc) the input data ``x``.
        
        Args:
            x (np.array): input data
        
        Returns:
            pre-processed data as np.array
        """
        
        y = x
        
        return y
        
    def predict(self, x):
        """Predict
        
        This function predicts ``x`` using ``self.pretrained_model``
        
        Args:
            x (np.array): input data
                            - image: shape is [1+``self.input_shape``], channel is [R, G, B]
        
        Return:
            prediction as dict object
        """
        
        tf_x = tf.convert_to_tensor(x, dtype=tf.uint8)
        prediction = self.pretrained_model(tf_x)
        
        return prediction
        
    def decode_predictions(self, preds, score_th=0.5):
        """Decode Predictions
        
        This function converts predictions to dict format
        
        Args:
            preds (np.array): predictions
            score_th (float): threshold to draw the bounding boxes
        
        Return:
            
        """
        
        np_preds = preds['detection_scores'].numpy()
        draw_preds = np_preds[0] >= score_th
        self.decoded_preds['num_detections'] = len(draw_preds[draw_preds])
        self.decoded_preds['detection_boxes'] = preds['detection_boxes'][0][draw_preds].numpy()
        self.decoded_preds['detection_classes'] = preds['detection_classes'][0][draw_preds].numpy()
        self.decoded_preds['detection_scores'] = preds['detection_scores'][0][draw_preds].numpy()

