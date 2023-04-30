"""Predictor for Keras

This file describe about the prediction process using Keras
"""

import cv2
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
        self.prediction = None
        
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
    
    def __init__(self, mlmodel, get_feature_map=False):
        """Constructor
        
        This function is the construction of predictor.
        
        Args:
            mlmodel (object): MlModel class
            get_feature_map (bool): If get feature map, set to True. Default is False.
        """
        
        # --- load config and set parameters ---
        config_path = Path(mlmodel.model_dir, 'config.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            task = config_data['inference_parameter']['model']['task']['value']
            self.input_shape = config_data['inference_parameter']['preprocessing']['input_shape']['value']
            self.norm_coef_a = config_data['inference_parameter']['preprocessing']['norm_coef_a']['value']
            self.norm_coef_b = config_data['inference_parameter']['preprocessing']['norm_coef_b']['value']
    
        # --- initialize ---
        self.get_feature_map = get_feature_map
        if (task in ['img_clf', 'table_clf']):
            self.task = 'classification'
        elif (task in ['img_reg', 'table_reg']):
            self.task = 'regression'
        else:
            # --- T.B.D ---
            self.task = 'object_detection'
        super().__init__(self.task)
        
        # --- load model ---
        trained_model_path = Path(mlmodel.model_dir, 'models', 'hdf5', 'model.h5')
        self.pretrained_model = keras.models.load_model(trained_model_path)
        self.pretrained_model.summary()
        
        if (self.get_feature_map):
            # --- If get feature map, re-define the model ---
            outputs = []
            for layer in self.pretrained_model.layers:
                if (layer.__class__.__name__ in ['Conv2D', 'Dense']):
                    outputs.append(layer.output)
            self.pretrained_model = keras.models.Model(inputs=self.pretrained_model.inputs, outputs=outputs)
        
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
        
        This function predicts ``x`` using ``self.pretrained_model``, converts predictions to dict format
        
        Args:
            x (np.array): input data
                            - image: shape is [[N]+``self.input_shape``], channel is [R, G, B]
        
        Returns:
            prediction as np.array
        """
        
        self.prediction = self.pretrained_model.predict(self.preprocess_input(x))
        
        if (self.get_feature_map):
            _preds = self.prediction[-1][0]
        else:
            _preds = self.prediction[0]
        
        top5_score = np.sort(_preds)[::-1][0:5]
        top5_class_id = np.argsort(_preds)[::-1][0:5]
        
        self.decoded_preds['class_id'] = top5_class_id
        self.decoded_preds['class_name'] = [f'class{i}' for i in top5_class_id]
        self.decoded_preds['score'] = top5_score

        
    def create_feature_map(self):
        """Create Freature Map
        
        This function converts ``self.prediction`` to the heatmap.
        
        """
        
        element_size = [5, 5]   # [H, W]
        offset = 5
        border = (2, 5)  # [H, W]
        
        # --- calculate min/max for normalization ---
        feature_min = self.prediction[0].min()
        feature_max = self.prediction[0].max()
        feature_ch_max = self.prediction[0].shape[-1]
        for feature in self.prediction[1:]:
            feature_min = min(feature_min, feature.min())
            feature_max = max(feature_max, feature.max())
            feature_ch_max = max(feature_ch_max, feature.shape[-1])
        layer_num = len(self.prediction)
        
        # --- calculate average and create feature map---
        feature_map_height = element_size[0] * feature_ch_max + border[0] * (feature_ch_max-1) + offset * 2
        feature_map_width = element_size[1] * layer_num + border[1] * (layer_num-1) + offset * 2
        feature_map = np.full([feature_map_height, feature_map_width, 3], 255, dtype=np.uint8)
        
        for _layer_num, feature in enumerate(self.prediction):
            feature_mean = feature.mean(axis=tuple(range(len(feature.shape)-1)))
            feature_norm = (feature_mean - feature_min) / (feature_max - feature_min)
            feature_map_vals = (feature_norm * 255).astype(int)
            
            for _ch, feature_map_val in enumerate(feature_map_vals):
                pos_x = offset + _layer_num*(element_size[1]+border[1])
                pos_y = offset + _ch*(element_size[0]+border[0])
                cv2.rectangle(feature_map, (pos_x, pos_y), (pos_x+element_size[0], pos_y+element_size[1]), (0, 0, 0), -1)
        
        return feature_map

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
#        self.pretrained_model = keras.models.Sequential(hub.KerasLayer(url))
#        self.pretrained_model.build(input_shape=[1, 512, 512, 3])
#        self.pretrained_model = keras.models.Sequential([hub.KerasLayer(url, input_shape=[512, 512, 3], dtype=tf.uint8)])

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

