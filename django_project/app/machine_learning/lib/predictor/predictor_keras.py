"""Predictor for Keras

This file describe about the prediction process using Keras
"""

import cv2
import json
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import logging

from pathlib import Path
from PIL import Image
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)

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
        self.get_feature_map = False
        self.feature_map_calc_range = 'Model-wise'
        
        if (task == 'classification'):
            self.decoded_preds = {
                'class_id': None,
                'class_name': None,
                'score': None,
            }
        elif ('object_detection' in task):
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
    
    def __init__(self, mlmodel, get_feature_map=False, feature_map_calc_range='Model-wise'):
        """Constructor
        
        This function is the construction of predictor.
        
        Args:
            mlmodel (object): MlModel class
            get_feature_map (bool): If get feature map, set to True. Default is False.
            feature_map_calc_range (string): calculation range of feature map
                - Model-wise
                - Layer-wise
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
        if (task in ['img_clf', 'table_clf']):
            self.task = 'classification'
        elif (task in ['img_reg', 'table_reg']):
            self.task = 'regression'
        else:
            # --- detection task ---
            if ('yolo' in task):
                self.task = 'object_detection_yolo'
            else:
                self.task = 'object_detection'
        super().__init__(self.task)
        self.get_feature_map = get_feature_map
        self.feature_map_calc_range = feature_map_calc_range
        
        # --- load model ---
        trained_model_path = Path(mlmodel.model_dir, 'models', 'hdf5', 'model.h5')
        custom_objects = None
        custom_object_path = Path(mlmodel.model_dir, 'models', 'custom_objects.pickle')
        if (custom_object_path.exists()):
            with open(custom_object_path, 'rb') as f:
                custom_objects = pickle.load(f)
        
        self.pretrained_model = keras.models.load_model(trained_model_path, custom_objects=custom_objects)
        self.pretrained_model.summary()
        
        if (self.get_feature_map):
            # --- If get feature map, re-define the model ---
            outputs = []
            for layer in self.pretrained_model.layers:
                if (layer.__class__.__name__ in ['Conv2D', 'Dense']):
                    outputs.append(layer.output)
            self.pretrained_model = keras.models.Model(inputs=self.pretrained_model.inputs, outputs=outputs)
        
        # --- for DEBUG ---
        self.yolo_nms_model = None
        
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
        
        This function predicts ``x`` using ``self.pretrained_model`` and converts predictions to dict format
        
        Args:
            x (np.array): input data
                            - image: shape is [[N]+``self.input_shape``], channel is [R, G, B]
        
        Returns:
            prediction as np.array
        """
        
        # --- inference ---
        #logging.info('-------------------------------------')
        #logging.info('[DEBUG]')
        #logging.info(f'  * x.shape: {x.shape}')
        #logging.info(f'  * self.norm_coef_a: {self.norm_coef_a}')
        #logging.info(f'  * self.norm_coef_b: {self.norm_coef_b}')
        #logging.info('-------------------------------------')
        self.prediction = self.pretrained_model.predict(self.preprocess_input(x))
        
        # --- post processing ---
        if (self.task == 'object_detection_yolo'):
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            
            def yolo_boxes(pred, anchors, classes):
                # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
                grid_size = pred.shape[1:3]
                box_xy, box_wh, objectness, class_probs = np.split(pred, (2, 4, 5), axis=-1)
                
                box_xy = sigmoid(box_xy)
                objectness = sigmoid(objectness)
                class_probs = sigmoid(class_probs)
                pred_box = np.concatenate((box_xy, box_wh), axis=-1)  # original xywh for loss

                # !!! grid[x][y] == (y, x)
                grid_x, grid_y = np.meshgrid(np.arange(grid_size[1]), np.arange(grid_size[0]))
                grid = np.stack((grid_x, grid_y), axis=-1)  # [gx, gy, 1, 2]
                grid = np.expand_dims(grid, axis=2)

                box_xy = (box_xy + grid) / np.array(grid_size).reshape(1, 1, 1, 2)
                box_wh = np.exp(box_wh) * anchors

                box_x1y1 = box_xy - box_wh / 2
                box_x2y2 = box_xy + box_wh / 2
                bbox = np.concatenate((box_x1y1, box_x2y2), axis=-1)

                return bbox, objectness, class_probs, pred_box
            
            def yolo_nms(outputs, anchors, masks, classes, yolo_max_boxes=100, yolo_iou_threshold=0.5, yolo_score_threshold=0.5):
                # boxes, conf, type
                b, c, t = [], [], []

                for o in outputs:
                    b.append(np.reshape(o[0], (np.shape(o[0])[0], -1, np.shape(o[0])[-1])))
                    c.append(np.reshape(o[1], (np.shape(o[1])[0], -1, np.shape(o[1])[-1])))
                    t.append(np.reshape(o[2], (np.shape(o[2])[0], -1, np.shape(o[2])[-1])))

                bbox = np.concatenate(b, axis=1)
                confidence = np.concatenate(c, axis=1)
                class_probs = np.concatenate(t, axis=1)

                # If we only have one class, do not multiply by class_prob (always 0.5)
                if classes == 1:
                    scores = confidence
                else:
                    scores = confidence * class_probs

                dscores = np.squeeze(scores, axis=0)
                #scores = np.reduce_max(dscores, axis=1)
                scores = np.apply_along_axis(max, 1, dscores)
                bbox = np.reshape(bbox, (-1, 4))
                classes = np.argmax(dscores, axis=1)

                selected_indices, selected_scores = non_max_suppression_with_scores(
                    boxes=bbox,
                    scores=scores,
                    max_output_size=yolo_max_boxes,
                    iou_threshold=yolo_iou_threshold,
                    score_threshold=yolo_score_threshold,
                    soft_nms_sigma=0.5
                )

                num_valid_nms_boxes = np.shape(selected_indices)[0]

                selected_indices = np.concatenate([selected_indices, np.zeros(yolo_max_boxes - num_valid_nms_boxes, np.int32)], 0)
                selected_scores = np.concatenate([selected_scores, np.zeros(yolo_max_boxes - num_valid_nms_boxes, np.float32)], -1)

                boxes = np.expand_dims(np.take(bbox, selected_indices, axis=0), axis=0)
                scores = np.expand_dims(selected_scores, axis=0)
                classes = np.expand_dims(np.take(classes, selected_indices), axis=0)
                valid_detections = np.expand_dims(num_valid_nms_boxes, axis=0)

                return boxes, scores, classes, valid_detections

            def non_max_suppression_with_scores(boxes, scores, max_output_size, iou_threshold, score_threshold, soft_nms_sigma):
                selected_indices = []

                # Sort boxes by scores in descending order
                sorted_indices = np.argsort(scores)[::-1]

                while sorted_indices.size > 0:
                    current_index = sorted_indices[0]
                    selected_indices.append(current_index)

                    if len(selected_indices) >= max_output_size:
                        break

                    # Compute IoU between the current box and the remaining boxes
                    current_box = boxes[current_index]
                    remaining_indices = sorted_indices[1:]

                    x1 = np.maximum(current_box[0], boxes[remaining_indices, 0])
                    y1 = np.maximum(current_box[1], boxes[remaining_indices, 1])
                    x2 = np.minimum(current_box[2], boxes[remaining_indices, 2])
                    y2 = np.minimum(current_box[3], boxes[remaining_indices, 3])

                    intersection_width = np.maximum(0.0, x2 - x1 + 1)
                    intersection_height = np.maximum(0.0, y2 - y1 + 1)
                    intersection_area = intersection_width * intersection_height

                    box_area = (current_box[2] - current_box[0] + 1) * (current_box[3] - current_box[1] + 1)
                    remaining_areas = (boxes[remaining_indices, 2] - boxes[remaining_indices, 0] + 1) * \
                                      (boxes[remaining_indices, 3] - boxes[remaining_indices, 1] + 1)
                    union_area = box_area + remaining_areas - intersection_area

                    iou = intersection_area / union_area

                    # Apply soft-NMS
                    weights = np.exp(-(iou * iou) / soft_nms_sigma)
                    scores[remaining_indices] *= weights

                    # Discard boxes with low scores
                    discard_indices = np.where(scores[remaining_indices] < score_threshold)[0]
                    sorted_indices = np.delete(sorted_indices, discard_indices + 1)

                selected_indices = np.array(selected_indices)
                selected_scores = scores[selected_indices]

                return selected_indices, selected_scores

            # As tensorflow lite doesn't support tf.size used in tf.meshgrid, 
            # we reimplemented a simple meshgrid function that use basic tf function.
            def _meshgrid(n_a, n_b):

                return [
                    tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
                    tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
                ]

            def tf_yolo_boxes(pred, anchors, classes):
                # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
                grid_size = tf.shape(pred)[1:3]
                box_xy, box_wh, objectness, class_probs = tf.split(
                    pred, (2, 2, 1, classes), axis=-1)

                box_xy = tf.sigmoid(box_xy)
                objectness = tf.sigmoid(objectness)
                class_probs = tf.sigmoid(class_probs)
                pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

                # !!! grid[x][y] == (y, x)
                grid = _meshgrid(grid_size[1],grid_size[0])
                grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

                box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
                    tf.cast(grid_size, tf.float32)
                box_wh = tf.exp(box_wh) * anchors

                box_x1y1 = box_xy - box_wh / 2
                box_x2y2 = box_xy + box_wh / 2
                bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

                return bbox, objectness, class_probs, pred_box


            def tf_yolo_nms(outputs, anchors, masks, classes, yolo_max_boxes=100, yolo_iou_threshold=0.5, yolo_score_threshold=0.5):
                # boxes, conf, type
                b, c, t = [], [], []

                for o in outputs:
                    b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
                    c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
                    t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

                bbox = tf.concat(b, axis=1)
                confidence = tf.concat(c, axis=1)
                class_probs = tf.concat(t, axis=1)

                # If we only have one class, do not multiply by class_prob (always 0.5)
                if classes == 1:
                    scores = confidence
                else:
                    scores = confidence * class_probs

                dscores = tf.squeeze(scores, axis=0)
                scores = tf.reduce_max(dscores,[1])
                bbox = tf.reshape(bbox,(-1,4))
                classes = tf.argmax(dscores,1)
                selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                    boxes=bbox,
                    scores=scores,
                    max_output_size=yolo_max_boxes,
                    iou_threshold=yolo_iou_threshold,
                    score_threshold=yolo_score_threshold,
                    soft_nms_sigma=0.5
                )
                
                num_valid_nms_boxes = tf.shape(selected_indices)[0]

                selected_indices = tf.concat([selected_indices,tf.zeros(yolo_max_boxes-num_valid_nms_boxes, tf.int32)], 0)
                selected_scores = tf.concat([selected_scores,tf.zeros(yolo_max_boxes-num_valid_nms_boxes,tf.float32)], -1)

                boxes=tf.gather(bbox, selected_indices)
                boxes = tf.expand_dims(boxes, axis=0)
                scores=selected_scores
                scores = tf.expand_dims(scores, axis=0)
                classes = tf.gather(classes,selected_indices)
                classes = tf.expand_dims(classes, axis=0)
                valid_detections=num_valid_nms_boxes
                valid_detections = tf.expand_dims(valid_detections, axis=0)

                return boxes, scores, classes, valid_detections

            
            from machine_learning.lib.trainer.tf_models.yolov3.models import yolo_anchors, yolo_anchor_masks
            
            output_0, output_1, output_2 = self.prediction
            
            yolo_nms_proc_tf = True
            if (yolo_nms_proc_tf):
                if (self.yolo_nms_model is None):
                    classes = 20
                    input_0 = Input(output_0.shape[1:], name='yolov3_nms_input_0')
                    input_1 = Input(output_1.shape[1:], name='yolov3_nms_input_1')
                    input_2 = Input(output_2.shape[1:], name='yolov3_nms_input_2')
                    
                    boxes_0 = Lambda(lambda x: tf_yolo_boxes(x, yolo_anchors[yolo_anchor_masks[0]], classes),
                                     name='yolo_boxes_0')(input_0)
                    boxes_1 = Lambda(lambda x: tf_yolo_boxes(x, yolo_anchors[yolo_anchor_masks[1]], classes),
                                     name='yolo_boxes_1')(input_1)
                    boxes_2 = Lambda(lambda x: tf_yolo_boxes(x, yolo_anchors[yolo_anchor_masks[2]], classes),
                                     name='yolo_boxes_2')(input_2)

                    outputs = Lambda(lambda x: tf_yolo_nms(x, yolo_anchors, yolo_anchor_masks, classes),
                                     name='yolov3_nms_output')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

                    self.yolo_nms_model = Model(inputs=[input_0, input_1, input_2], outputs=outputs, name='yolov3_nms')
                    self.yolo_nms_model.summary(print_fn=logging.info)
                
                self.prediction = self.yolo_nms_model.predict([output_0, output_1, output_2])
            else:
                
                #logging.info('-------------------------------------')
                #logging.info('[DEBUG]')
                #logging.info(f'  * output_0.shape: {output_0.shape}')
                #logging.info(f'  * output_1.shape: {output_1.shape}')
                #logging.info(f'  * output_2.shape: {output_2.shape}')
                #logging.info('-------------------------------------')
                boxes_0 = yolo_boxes(output_0, yolo_anchors[yolo_anchor_masks[0]], 20)
                boxes_1 = yolo_boxes(output_1, yolo_anchors[yolo_anchor_masks[1]], 20)
                boxes_2 = yolo_boxes(output_2, yolo_anchors[yolo_anchor_masks[2]], 20)
                
                self.prediction = yolo_nms((boxes_0[:3], boxes_1[:3], boxes_2[:3]), yolo_anchors, yolo_anchor_masks, 20)

            boxes, scores, classes, valid_detections = self.prediction
            #logging.info('-------------------------------------')
            #logging.info('[DEBUG]')
            #logging.info(f'  * len(self.prediction): {len(self.prediction)}')
            #logging.info(f'  * valid_detections: {valid_detections}')
            #if (len(valid_detections) > 0):
            #    logging.info(f'  * box: {boxes[0]}')
            #    logging.info(f'  * scores: {scores[0]}')
            #    logging.info(f'  * classes: {classes[0]}')
            #
            #logging.info('-------------------------------------')
            
        self.decoded_preds['num_detections'] = valid_detections[0]
        self.decoded_preds['detection_boxes'] = np.array(boxes[0][0:valid_detections[0]])[:, [1, 0, 3, 2]]  # [x1, y1, x2, y2]->[y1, x1, y2, x1]
        self.decoded_preds['detection_classes'] = np.array(classes[0][0:valid_detections[0]])
        self.decoded_preds['detection_scores'] = np.array(scores[0][0:valid_detections[0]])
        
    def create_feature_map(self):
        """Create Freature Map
        
        This function converts ``self.prediction`` to the heatmap.
        
        """
        
        element_size = [5, 5]   # [H, W]
        offset = 5
        border = (2, 5)  # [H, W]
        
        # --- calculate min/max for normalization ---
        if (self.feature_map_calc_range == 'Model-wise'):
            feature_min = self.prediction[0].min()
            feature_max = self.prediction[0].max()
            feature_ch_max = self.prediction[0].shape[-1]
            for feature in self.prediction[1:]:
                feature_min = min(feature_min, feature.min())
                feature_max = max(feature_max, feature.max())
                feature_ch_max = max(feature_ch_max, feature.shape[-1])
            layer_num = len(self.prediction)
            
            feature_min = [feature_min for _ in range(layer_num)]
            feature_max = [feature_max for _ in range(layer_num)]
        else:
            feature_min = [self.prediction[0].min()]
            feature_max = [self.prediction[0].max()]
            feature_ch_max = self.prediction[0].shape[-1]
            for feature in self.prediction[1:]:
                feature_min.append(feature.min())
                feature_max.append(feature.max())
                feature_ch_max = max(feature_ch_max, feature.shape[-1])
            layer_num = len(self.prediction)
        
        # --- calculate average and create feature map---
        feature_map_height = element_size[0] * feature_ch_max + border[0] * (feature_ch_max-1) + offset * 2
        feature_map_width = element_size[1] * layer_num + border[1] * (layer_num-1) + offset * 2
        feature_map = np.full([feature_map_height, feature_map_width, 3], 255, dtype=np.uint8)
        
        for _layer_num, feature in enumerate(self.prediction):
            feature_mean = feature.mean(axis=tuple(range(len(feature.shape)-1)))
            feature_norm = (feature_mean - feature_min[_layer_num]) / (feature_max[_layer_num] - feature_min[_layer_num])
            feature_map_vals = (feature_norm * 255).astype(int)
            
            for _ch, feature_map_val in enumerate(feature_map_vals):
                pos_x = offset + _layer_num*(element_size[1]+border[1])
                pos_y = offset + _ch*(element_size[0]+border[0])
                
                color = np.array([feature_map_val, feature_map_val, 0]).tolist()
                cv2.rectangle(feature_map, (pos_x, pos_y), (pos_x+element_size[0], pos_y+element_size[1]), color, -1)
        
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
        
        This function predicts ``x`` using ``self.pretrained_model`` and converts predictions to dict format
        
        Args:
            x (np.array): input data
                            - image: shape is [[N]+``self.input_shape``], channel is [R, G, B]
        
        Return:
            prediction as np.array
        """
        
        preprocessing_input_ = keras.applications.resnet50.preprocess_input
        self.prediction = self.pretrained_model.predict(preprocessing_input_(x))
        
        decode_predictions = keras.applications.resnet50.decode_predictions
        decoded_preds_ = decode_predictions(self.prediction, top=5)[0]
        
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
        
    def predict(self, x, score_th=0.5):
        """Predict
        
        This function predicts ``x`` using ``self.pretrained_model`` and converts predictions to dict format
        
        Args:
            x (np.array): input data
                            - image: shape is [1+``self.input_shape``], channel is [R, G, B]
            score_th (float): threshold to draw the bounding boxes
        
        Return:
            prediction as dict object
        """
        
        tf_x = tf.convert_to_tensor(x, dtype=tf.uint8)
        self.prediction = self.pretrained_model(tf_x)
        
        np_preds = self.prediction['detection_scores'].numpy()
        draw_preds = np_preds[0] >= score_th
        self.decoded_preds['num_detections'] = len(draw_preds[draw_preds])
        self.decoded_preds['detection_boxes'] = self.prediction['detection_boxes'][0][draw_preds].numpy()
        self.decoded_preds['detection_classes'] = self.prediction['detection_classes'][0][draw_preds].numpy()
        self.decoded_preds['detection_scores'] = self.prediction['detection_scores'][0][draw_preds].numpy()
        

