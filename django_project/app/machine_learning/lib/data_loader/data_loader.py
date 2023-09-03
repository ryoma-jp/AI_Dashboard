#! -*- coding: utf-8 -*-

"""Data Loader

This file is written about the dataset loading.
"""

#---------------------------------
# Import modules
#---------------------------------
import os
import logging
import numpy as np
import pandas as pd
import requests
import tarfile
import gzip
import json
import hashlib
import shutil

import tensorflow as tf

from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
from lxml import etree as lxml_etree

from machine_learning.lib.utils.utils import zip_extract, download_file, safe_extract_tar, parse_xml
from machine_learning.lib.utils.preprocessor import image_preprocess

#---------------------------------
# Functions
#---------------------------------
def build_tf_example(annotation, class_map, imagefile_dir=''):
    """Build TensorFlow Examples
    
    Build TensorFlow examples
    
    Args:
        annotation (dict): annotations as dict object
                           <dict format>
                               {
                                   'filename': <image file name>,
                                   'size': {
                                       'width': <image width>,
                                       'height': <image height>,
                                       'depth': <channels>,
                                   },
                                   'object': [
                                       {
                                           'bndbox': {
                                               'xmin': <x pos of left top>,
                                               'ymin': <y pos of left top>,
                                               'xmax': <x pos of right bottom>,
                                               'xmin': <x pos of right bottom>,
                                           },
                                           'name': <class name>,
                                       },
                                       ...
                                   ]
                               }
        class_map (dict): mapping data between the class name and the class id as dict object
                          <dict format>
                              {
                                  <class name>: <class id>,
                                  ...
                              }
        imagefile_dir (pathlib.Path): directory of image file
    """
    img_path = Path(imagefile_dir, annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    bbox_width = []
    bbox_height = []
    bbox = []
    classes = []
    classes_text = []
    classes_name = []
    if 'object' in annotation:
        for obj in annotation['object']:
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            bbox_width.append(float(obj['bndbox']['xmax']) / width - float(obj['bndbox']['xmin']) / width)
            bbox_height.append(float(obj['bndbox']['ymax']) / height - float(obj['bndbox']['ymin']) / height)
            bbox.append([
                float(obj['bndbox']['xmin']),
                float(obj['bndbox']['ymin']),
                float(obj['bndbox']['xmax']) - float(obj['bndbox']['xmin']),
                float(obj['bndbox']['ymax']) - float(obj['bndbox']['ymin']),
            ])
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])
            classes_name.append(obj['name'])

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation['filename'].encode('utf8')])),
        'source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation['filename'].encode('utf8')])),
        'key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_path.suffix.lstrip('.').encode('utf8')])),
        'target/bbox/x': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'target/bbox/y': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'target/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'target/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'target/bbox/width': tf.train.Feature(float_list=tf.train.FloatList(value=bbox_width)),
        'target/bbox/height': tf.train.Feature(float_list=tf.train.FloatList(value=bbox_height)),
        'target/class_text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'target/class_id': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))

    info_target = {
        'class_id': classes,
        'category_name': classes_name,
        'bbox': bbox,
    }
    
    return example, info_target

def load_dataset_from_tfrecord(tfrecord, class_name_file, img_size):
    """Load Dataset from TFRecord
    
    Load tfrecord and output dataset as TensorFlow dataset
    
    Args:
        tfrecord: dataset as tfrecord format
        class_name_file: class name file path
        img_size: image size [img_size, img_size]
    
    Returns:
        dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
    """
    
    def _parse_tfrecord(tfrecord, class_table, size):
        # --- parse and scaling ---
        features = {
            'encoded': tf.io.FixedLenFeature([], tf.string),
            'target/bbox/x': tf.io.VarLenFeature(tf.float32),
            'target/bbox/y': tf.io.VarLenFeature(tf.float32),
            'target/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'target/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'target/class_text': tf.io.VarLenFeature(tf.string),
        }
        x = tf.io.parse_single_example(tfrecord, features)
        x_train = tf.image.decode_jpeg(x['encoded'], channels=3)
        x_train = tf.image.resize(x_train, (size, size))
        
        # --- parse annotations ---
        class_names = tf.sparse.to_dense(x['target/class_text'], default_value='')
        labels = tf.cast(class_table.lookup(class_names), tf.float32)
        y_train = tf.stack([tf.sparse.to_dense(x['target/bbox/x']),
                        tf.sparse.to_dense(x['target/bbox/y']),
                        tf.sparse.to_dense(x['target/bbox/xmax']),
                        tf.sparse.to_dense(x['target/bbox/ymax']),
                        labels], axis=1)
        paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]]
        print(paddings)
        y_train = tf.pad(y_train, paddings)
        
        return x_train, y_train
    
    # --- load class table and tfrecord ---
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_name_file, tf.string, 0, tf.int64, -1, delimiter="\n"), -1)
    dataset = tf.data.Dataset.list_files(tfrecord).flat_map(tf.data.TFRecordDataset)
    
    # --- convert format to tf.data.Dataset and return---
    return dataset.map(lambda x: _parse_tfrecord(x, class_table, img_size))
        
#---------------------------------
# Class
#---------------------------------
class DataLoader():
    """DataLoader
    
    Base class for dataset loading
    
    Attributes:
        one_hot (bool): If the ground truth is onehot, set to True
        output_dims (int): Output dimensions
        verified (bool): Verified status of Web app (True: verified, False: not verified)
        train_x (numpy.ndarray): Input data for training
        train_y (numpy.ndarray): Ground truth for training
        validation_x (numpy.ndarray): Input data for validation
        validation_y (numpy.ndarray): Ground truth for validation
        test_x (numpy.ndarray): Input data for test
        test_y (numpy.ndarray): Ground truth for test
        dataset_type (str): Type of dataset
            - 'img_clf': Image Classification
            - 'img_det': Image Detection
            - 'img_reg': Image Regression
            - 'table_clf': Table data Classification
            - 'table_reg': Table data Regression
        input_distributions (dict): Distributions of input features
        target_distributions (dict): Distributions of targets
        statistic_keys (list): Feature name list of distribution culculation
        preprocessing_params (dict): Parameters of preprocessing
    
    """
    def __init__(self):
        """Constructor
        
        Constructor
        """
        self.train_x = None
        self.train_y = None
        self.validation_x = None
        self.validation_y = None
        self.test_x = None
        self.test_y = None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        
        self.one_hot = True
        self.output_dims = -1
        self.verified = False
        self.dataset_type = None
        self.input_distributions = None
        self.target_distributions = None
        self.statistic_keys = []
        self.preprocessing_params = {
            'norm_coef': [0.0, 1.0],
        }
        
    def preprocessing(self, norm_mode='none'):
        """Pre-processing
        
        Data pre-processing.
        Pre-processing methods are below.
            - normalization (or standardization)
        
        Args:
            norm_mode (str): Specify the normalization method
                - 'none': Trough
                - 'max': Divide max value(=255)
                - 'max-min': Normalization to value range that is from 0.0 to 1.0
                - 'z-score': Standardization
        
        Returns:
            train, validation and test dataset are preprocessed.
            return the ``y`` value is for future update, example for add scaling or cropping.
        
        Note:
            - [Standardization and Normalization](https://aiacademy.jp/texts/show/?id=555)
        """
        # --- Initialize ---
        preprocessed_train_x = None
        preprocessed_train_y = self.train_y
        preprocessed_validation_x = None
        preprocessed_validation_y = self.validation_y
        preprocessed_test_x = None
        preprocessed_test_y = self.test_y
        
        # --- Normalization process ---
        if (norm_mode == 'none'):
            self.preprocessing_params['norm_coef'] =  [0.0, 1.0]
        elif (norm_mode == 'max'):
            self.preprocessing_params['norm_coef'] = [0.0, 255.0]
        elif (norm_mode == 'max-min'):
            train_min = np.min(self.train_x)
            train_diff = np.max(self.train_x) - np.min(self.train_x)
            train_diff = np.clip(train_diff, np.finfo(float).eps, None)
            
            self.preprocessing_params['norm_coef'] = [train_min, train_diff]
        elif (norm_mode == 'z-score'):
            self.preprocessing_params['norm_coef'] = [np.mean(self.train_x), np.std(self.train_x)]
        else:
            logging.info(f'[WARNING] Unknown data normalization mode: {mode}')
            self.preprocessing_params['norm_coef'] = [0.0, 1.0]
        
        preprocessed_train_x = image_preprocess(self.train_x, self.preprocessing_params['norm_coef'])
        if (self.validation_x is not None):
            preprocessed_validation_x = image_preprocess(self.validation_x, self.preprocessing_params['norm_coef'])
        if (self.test_x is not None):
            preprocessed_test_x = image_preprocess(self.test_x, self.preprocessing_params['norm_coef'])
        
        return preprocessed_train_x, preprocessed_train_y, \
               preprocessed_validation_x, preprocessed_validation_y, \
               preprocessed_test_x, preprocessed_test_y
        
    def split_train_val(self, validation_split):
        """split_train_val
        
        Split dataset training dataset and validation dataset
        
        Args:
            validation_split (float): Ratio of validation data
        
        """
        idx = np.arange(len(self.train_x))
        np.random.shuffle(idx)
        
        if ((1.0 - validation_split) == 1.0):
            self.validation_x = None
            self.validation_y = None
        elif ((1.0 - validation_split) == 0.0):
            self.validation_x = self.train_x
            self.validation_y = self.train_y
            self.train_x = None
            self.train_y = None
        else:
            validation_index = int(len(self.train_x) * (1.0-validation_split))
            self.validation_x = self.train_x[validation_index:]
            self.validation_y = self.train_y[validation_index:]
            self.train_x = self.train_x[0:validation_index]
            self.train_y = self.train_y[0:validation_index]
        
        return
    
    def train_valid_test_split(self, validation_size=0.2, test_size=0.3):
        """Split Dataset
        
        This function split dataset from train to train, validation and test.
        
        Args:
            validation_size (float): validation data rate (0.0 <= validation_size < 1.0)
            test_size (float): validation data rate (0.0 <= validation_size < 1.0)
        """
        
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.train_x, self.train_y, test_size=test_size, random_state=42)
        
        self.train_x, self.validation_x, self.train_y, self.validation_y = train_test_split(
            self.train_x, self.train_y, test_size=validation_size, random_state=42)
        
    
    def convert_label_encoding(self, one_hot=True):
        """Convert Label Encoding
        
        Convert the ground truth represenation onehot and index
        
        Args:
            one_hot (bool): True: index->onehot, False: onehot->index
        """
        
        if ((not self.one_hot) and (one_hot)):
            identity = np.eye(self.output_dims, dtype=np.int)
            self.train_y = np.array([identity[i] for i in self.train_y])
            
            if (self.validation_y is not None):
                self.validation_y = np.array([identity[i] for i in self.validation_y])
            
            if (self.test_y is not None):
                self.test_y = np.array([identity[i] for i in self.test_y])
        elif ((self.one_hot) and (not one_hot)):
            self.train_y = self.train_y.argmax(axis=1)
            
            if (self.validation_y is not None):
                self.validation_y = self.validation_y.argmax(axis=1)
            
            if (self.test_y is not None):
                self.test_y = self.test_y.argmax(axis=1)
            
        self.one_hot = one_hot
        
        return
    
    def data_analysis(self):
        """Data Analysis
        
        This function analylizies dataset.
        """
        
        # --- Calculate input distribution ---
        if (self.dataset_type in ['img_clf', 'img_reg']):
            # --- Image data ---
            self.input_distributions = None
        elif (self.dataset_type in ['table_clf', 'table_reg']):
            # --- Table data ---
            
            # --- Initialize ---
            self.input_distributions = {}
            self.statistic_keys.append("Input Distributions")
            
            # --- Set bins
            bins = 20
            
            # --- Calculate input distribution ---
            if (self.train_x is not None):
                self.input_distributions['train'] = {}
                
                for _key in self.train_x.columns:
                    self.input_distributions['train'][_key] = {}
                    hist_y, hist_x = np.histogram(self.train_x[_key], bins=bins)
                    hist_x = np.round(hist_x, decimals=2)
                    self.input_distributions['train'][_key]['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                    self.input_distributions['train'][_key]['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.input_distributions['train'] = None
                
            if (self.validation_x is not None):
                self.input_distributions['validation'] = {}
                
                for _key in self.validation_x.columns:
                    self.input_distributions['validation'][_key] = {}
                    hist_y, hist_x = np.histogram(self.validation_x[_key], bins=bins)
                    hist_x = np.round(hist_x, decimals=2)
                    self.input_distributions['validation'][_key]['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                    self.input_distributions['validation'][_key]['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.input_distributions['validation'] = None
                
            if (self.test_x is not None):
                self.input_distributions['test'] = {}
                
                for _key in self.test_x.columns:
                    self.input_distributions['test'][_key] = {}
                    hist_y, hist_x = np.histogram(self.test_x[_key], bins=bins)
                    hist_x = np.round(hist_x, decimals=2)
                    self.input_distributions['test'][_key]['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                    self.input_distributions['test'][_key]['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.input_distributions['test'] = None
                
            
        else:
            # --- Unknown dataset type ---
            self.input_distributions = None
        
        # --- Calculate target distribution ---
        if (self.target_distributions is None):
            # --- Initialize ---
            self.target_distributions = {}
            self.statistic_keys.append("Target Distributions")
            
            # --- Set bins ---
            if (self.dataset_type in ['img_clf', 'table_clf']):
                # --- Set sequence of scalars to bins if dataset is the classification task ---
                bins = np.unique(self.train_y)
                if (self.validation_y is not None):
                    _targets = np.unique(self.validation_y)
                    _targets = np.hstack((_targets, max(_targets)+1))
                    if (len(bins) < len(_targets)):
                        bins = _targets
                if (self.test_y is not None):
                    _targets = np.unique(self.test_y)
                    if (len(bins) < len(_targets)):
                        bins = _targets
            else:
                # --- Set scalar to bins if dataset is the regression task ---
                bins = 20
            
            # --- Create histograms ---
            if (self.train_y is not None):
                self.target_distributions['train'] = {}
                hist_y, hist_x = np.histogram(self.train_y, bins=bins)
                if (self.dataset_type in ['img_reg', 'table_reg']):
                    hist_x = np.round(hist_x, decimals=2)
                self.target_distributions['train']['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                self.target_distributions['train']['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.target_distributions['train'] = None
            
            if (self.validation_y is not None):
                self.target_distributions['validation'] = {}
                hist_y, hist_x = np.histogram(self.validation_y, bins=bins)
                if (self.dataset_type in ['img_reg', 'table_reg']):
                    hist_x = np.round(hist_x, decimals=2)
                self.target_distributions['validation']['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                self.target_distributions['validation']['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.target_distributions['validation'] = None
            
            if (self.test_y is not None):
                self.target_distributions['test'] = {}
                hist_y, hist_x = np.histogram(self.test_y, bins=bins)
                if (self.dataset_type in ['img_reg', 'table_reg']):
                    hist_x = np.round(hist_x, decimals=2)
                self.target_distributions['test']['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                self.target_distributions['test']['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.target_distributions['test'] = None
            
class DataLoaderCIFAR10(DataLoader):
    """DataLoaderCIFAR10
    
    DataLoader class for CIFAR-10 dataset
    """
    
    def __init__(self, dataset_dir, validation_split=0.0, flatten=False, one_hot=False, download=False):
        """Constructor
        
        Constructor
        
        Args:
            dataset_dir (string): dataset directory
            validation_split (float): ratio of validation data
            flatten (bool): [T.B.D] If input shape is vector([N, :]), set to True
            one_hot (bool): If the ground truth is onehot, set to True
            download (bool): If the dataset downloads from Web, set to True
        """
        
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        # --- initialize super class ---
        super().__init__()
        self.one_hot = one_hot
        self.verified = True
        
        # --- download dataset and extract ---
        if (download):
            logging.info(f'[DataLoaderCIFAR10] {dataset_dir}')
            os.makedirs(dataset_dir, exist_ok=True)
            if (not Path(dataset_dir, 'cifar-10-batches-py').exists()):
                url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
                save_file = download_file(url, dataset_dir)
                
                with tarfile.open(save_file) as tar:
                    # --- CVE-2007-4559 start ---
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = Path(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    safe_extract(tar, path=dataset_dir)
                    # --- CVE-2007-4559 end ---
            else:
                logging.info('CIFAR-10 dataset is exists (Skip Download)')
        dataset_dir = Path(dataset_dir, 'cifar-10-batches-py')
            
        # --- load training data ---
        train_data_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        dict_data = unpickle(Path(dataset_dir, train_data_list[0]))
        train_x = dict_data[b'data']
        train_y = dict_data[b'labels'].copy()
        for train_data in train_data_list[1:]:
            dict_data = unpickle(Path(dataset_dir, train_data))
            train_x = np.vstack((train_x, dict_data[b'data']))
            train_y = np.hstack((train_y, dict_data[b'labels']))
        
        # --- load test data ---
        test_data = "test_batch"
        dict_data = unpickle(Path(dataset_dir, test_data))
        test_x = dict_data[b'data']
        test_y = dict_data[b'labels'].copy()
        
        # --- transpose: [N, C, H, W] -> [N, H, W, C] ---
        self.train_x = train_x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.test_x = test_x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # --- labels ---
        if (self.one_hot):
            identity = np.eye(10, dtype=np.int)
            self.train_y = np.array([identity[i] for i in train_y])
            self.test_y = np.array([identity[i] for i in test_y])
        else:
            self.train_y = np.array(train_y)
            self.test_y = np.array(test_y)
        
        # --- split dataset training data and validation data ---
        self.split_train_val(validation_split)
        
        # --- save output dimension ---
        self.output_dims = 10
        
        # --- set task to image classification ---
        self.dataset_type = 'img_clf'
        
        return
        
class DataLoaderMNIST(DataLoader):
    """DataLoaderMNIST
    
    DataLoader class for MNIST dataset
    """
    
    def __init__(self, dataset_dir, validation_split=0.0, flatten=False, one_hot=False, download=False):
        """Constructor
        
        Constructor
        
        Args:
            dataset_dir (string): dataset directory
            validation_split (float): ratio of validation data
            flatten (bool): [T.B.D] If input shape is vector([N, :]), set to True
            one_hot (bool): If the ground truth is onehot, set to True
            download (bool): If the dataset downloads from Web, set to True
        """
        
        # --- initialize super class ---
        super().__init__()
        self.one_hot = one_hot
        self.verified = True
        
        # --- download dataset and extract ---
        if (download):
            logging.info(f'[DataLoaderMNIST] {dataset_dir}')
            os.makedirs(dataset_dir, exist_ok=True)
            mnist_files = [
                'train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz'
            ]
            
            for mnist_file in mnist_files:
                if (not Path(dataset_dir, mnist_file).exists()):
                    url = 'http://yann.lecun.com/exdb/mnist/' + mnist_file
                    save_file = download_file(url, dataset_dir)
                    
                    with gzip.open(save_file, 'rb') as gz:
                        gz_content = gz.read()
                    
                    save_file = Path(dataset_dir, mnist_file[:-3])
                    with open(save_file, 'wb') as f:
                        f.write(gz_content)
                    
                else:
                    logging.info(f'{mnist_file} is exists (Skip Download)')
            
        # --- load training data ---
        f = open(Path(dataset_dir, 'train-images-idx3-ubyte'))
        byte_data = np.fromfile(f, dtype=np.uint8)
        
        n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
        img_h = (byte_data[8] << 24) | (byte_data[9] << 16) | (byte_data[10] << 8) | (byte_data[11])
        img_w = (byte_data[12] << 24) | (byte_data[13] << 16) | (byte_data[14] << 8) | (byte_data[15])
        
        if (flatten):
            self.train_x = byte_data[16:].reshape(n_items, -1)
        else:
            self.train_x = byte_data[16:].reshape(n_items, img_h, img_w, 1)
        
        # --- load training label ---
        f = open(Path(dataset_dir, 'train-labels-idx1-ubyte'))
        byte_data = np.fromfile(f, dtype=np.uint8)
        
        n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
        
        self.train_y = byte_data[8:]
        if (self.one_hot):
            identity = np.eye(10, dtype=np.int)
            self.train_y = np.array([identity[i] for i in self.train_y])
        
        # --- load test data ---
        f = open(Path(dataset_dir, 't10k-images-idx3-ubyte'))
        byte_data = np.fromfile(f, dtype=np.uint8)
        
        n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
        img_h = (byte_data[8] << 24) | (byte_data[9] << 16) | (byte_data[10] << 8) | (byte_data[11])
        img_w = (byte_data[12] << 24) | (byte_data[13] << 16) | (byte_data[14] << 8) | (byte_data[15])
        
        if (flatten):
            self.test_x = byte_data[16:].reshape(n_items, -1)
        else:
            self.test_x = byte_data[16:].reshape(n_items, img_h, img_w, 1)
        
        # --- load test label ---
        f = open(Path(dataset_dir, 't10k-labels-idx1-ubyte'))
        byte_data = np.fromfile(f, dtype=np.uint8)
        
        n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
        
        self.test_y = byte_data[8:]
        if (self.one_hot):
            self.test_y = np.array([identity[i] for i in self.test_y])
        
        # --- split dataset training data and validation data ---
        self.split_train_val(validation_split)
        
        # --- save output dimension ---
        self.output_dims = 10
        
        # --- set task to image classification ---
        self.dataset_type = 'img_clf'
        
        return
    
class DataLoaderCOCO2017(DataLoader):
    """DataLoaderCOCO2017
    
    DataLoader class for COCO2017 dataset.
    In this sample code, the validation data is used as the test data, and the training data is split the training data and validation data.
    """
    
    def __init__(self, dataset_dir, validation_split=0.0, flatten=False, one_hot=False, download=False, model_input_size=224):
        """Constructor
        
        Constructor
        
        Args:
            dataset_dir (string): dataset directory
            validation_split (float): ratio of validation data
            flatten (bool): [T.B.D] If input shape is vector([N, :]), set to True
            one_hot (bool): If the ground truth is onehot, set to True
            download (bool): If the dataset downloads from Web, set to True
        """
        
        def _get_instances(instances_json):
            def _get_licenses(x, df_licenses=None):
                license = df_licenses[df_licenses['id']==x['license']].iloc[0]
                
                dict_rename = dict(zip(license.index, [f'license_{index}' for index in license.index]))
                license.rename(dict_rename, inplace=True)
                
                return license
            
            def _get_instances_annotations(x, df_images=None, df_categories=None):
                image = df_images[df_images['id']==x['image_id']].iloc[0]
                image.rename({'id': 'image_id'}, inplace=True)
                
                x.drop(index='image_id', inplace=True)
                x.rename({'id': 'instance_id'}, inplace=True)
                
                category = df_categories[df_categories['id']==x['category_id']].iloc[0]
                category.rename({'id': 'category_id', 'name': 'category_name'}, inplace=True)
                
                x.drop(index='category_id', inplace=True)
                
                annotation = pd.concat([image, x, category])
                
                return annotation
            
            with open(instances_json, 'r') as f:
                instance_data = json.load(f)
            
            df_instances = pd.DataFrame(instance_data["images"])
            
            logging.info(f'instance_data.keys(): {instance_data.keys()}')
            
            df_instances_licenses = df_instances.apply(_get_licenses, axis=1, df_licenses=pd.DataFrame(instance_data["licenses"]))
            df_instances = pd.concat([df_instances, df_instances_licenses], axis=1)
            df_instances.drop(columns=['license'], inplace=True)
            
            df_instances_annotations = pd.DataFrame(instance_data['annotations'])
            df_instances_categories = pd.DataFrame(instance_data['categories'])
            df_instances = df_instances_annotations.apply(
                               _get_instances_annotations,
                               axis=1,
                               df_images=df_instances,
                               df_categories=df_instances_categories)
            
            return df_instances
        
        # --- initialize super class ---
        super().__init__()
        self.one_hot = one_hot
        self.verified = True
        
        # --- download dataset and extract ---
        if (download):
            logging.info(f'[DataLoaderCOCO2017] {dataset_dir}')
            os.makedirs(dataset_dir, exist_ok=True)
            
            # --- download trainval2017 ---
            if (not Path(dataset_dir, 'annotations_trainval2017.zip').exists()):
                logging.info('downloading annotations_trainval2017.zip')
                url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
                save_file = download_file(url, dataset_dir)
                zip_extract(save_file, dataset_dir)
            else:
                logging.info('annotations_trainval2017.zip is exists (Skip Download)')
            
            # --- download train2017 ---
            if (not Path(dataset_dir, 'train2017.zip').exists()):
                logging.info('downloading train2017.zip')
                url = 'http://images.cocodataset.org/zips/train2017.zip'
                save_file = download_file(url, dataset_dir)
                zip_extract(save_file, dataset_dir)
                
            else:
                logging.info('train2017.zip is exists (Skip Download)')

            # --- download val2017 ---
            if (not Path(dataset_dir, 'val2017.zip').exists()):
                logging.info('downloading val2017.zip')
                url = 'http://images.cocodataset.org/zips/val2017.zip'
                save_file = download_file(url, dataset_dir)
                zip_extract(save_file, dataset_dir)
                
            else:
                logging.info('val2017.zip is exists (Skip Download)')
        
        # --- load annotations(instances) ---
        instances_json = Path(dataset_dir, 'annotations', 'instances_train2017.json')
        df_instances_train_val = _get_instances(instances_json)
        
        validation_split = np.clip(validation_split, 0, 0.5)
        if (validation_split == 0.0):
            self.df_instances_train = df_instances_train_val
            self.df_instances_train.to_csv(Path(dataset_dir, 'instances_train.csv'))
            self.df_instances_validation = None
        else:
            image_ids_train_val = df_instances_train_val['image_id'].unique()
            
            n_train = int(len(image_ids_train_val) * (1.0 - validation_split))
            train_ids = image_ids_train_val[0:n_train]
            self.df_instances_train = df_instances_train_val[df_instances_train_val['image_id'].map(lambda x: x in train_ids)]
            self.df_instances_train.to_csv(Path(dataset_dir, 'instances_train.csv'))
            
            valid_ids = image_ids_train_val[n_train::]
            self.df_instances_validation = df_instances_train_val[df_instances_train_val['image_id'].map(lambda x: x in valid_ids)]
            self.df_instances_validation.to_csv(Path(dataset_dir, 'instances_validation.csv'))
        
        instances_json = Path(dataset_dir, 'annotations', 'instances_val2017.json')
        self.df_instances_test = _get_instances(instances_json)
        self.df_instances_test.to_csv(Path(dataset_dir, 'instances_test.csv'))
        
        # --- save output dimension ---
        self.output_dims = self.df_instances_train['category_id'].nunique()
        
        # --- set task to image classification ---
        self.dataset_type = 'img_det'
        
        # --- T.B.D ---
        #   * too many the memory necessary
        train_x = []
        train_file_names = self.df_instances_train['file_name'].unique()
        for file_name in train_file_names[0:min(len(train_file_names), 100)]:
            img = Image.open(Path(dataset_dir, 'train2017', file_name)).convert('RGB')
            img = img.resize([model_input_size, model_input_size], Image.BILINEAR)
            train_x.append(np.array(img.getdata()).reshape(model_input_size, model_input_size, 3).tolist())
            
        self.train_x = np.array(train_x)
        self.train_y = None
        
        validation_x = []
        validation_file_names = self.df_instances_validation['file_name'].unique()
        for file_name in validation_file_names[0:min(len(validation_file_names), 20)]:
            img = Image.open(Path(dataset_dir, 'train2017', file_name)).convert('RGB')
            img = img.resize([model_input_size, model_input_size], Image.BILINEAR)
            validation_x.append(np.array(img.getdata()).reshape(model_input_size, model_input_size, 3).tolist())
        self.validation_x = np.array(validation_x)
        self.validation_y = None
        
        test_x = []
        test_file_names = self.df_instances_test['file_name'].unique()
        for file_name in test_file_names[0:min(len(test_file_names), 20)]:
            img = Image.open(Path(dataset_dir, 'val2017', file_name)).convert('RGB')
            img = img.resize([model_input_size, model_input_size], Image.BILINEAR)
            test_x.append(np.array(img.getdata()).reshape(model_input_size, model_input_size, 3).tolist())
        self.test_x = np.array(test_x)
        self.test_y = None
        
        return
        
class DataLoaderPascalVOC2012(DataLoader):
    """DataLoaderPascalVOC2012
    
    DataLoader class for PascalVOC 2012 dataset.
    In this sample code, the validation data is used as the test data, and the training data is split the training data and validation data.
    """
    
    def __init__(self, dataset_dir, validation_split=0.0, flatten=False, one_hot=False, download=False, model_input_size=416):
        """Constructor
        
        Constructor
        
        Args:
            dataset_dir (string): dataset directory
            validation_split (float): ratio of validation data
            flatten (bool): [T.B.D] If input shape is vector([N, :]), set to True
            one_hot (bool): If the ground truth is onehot, set to True
            download (bool): If the dataset downloads from Web, set to True
            model_input_size (int): image size using to scale to input to the model
        """
        
        # --- initialize super class ---
        super().__init__()
        self.one_hot = one_hot
        self.verified = True
        
        # --- download dataset and extract ---
        if (download):
            logging.info(f'[DataLoaderPascalVOC2012] {dataset_dir}')
            os.makedirs(dataset_dir, exist_ok=True)
            
            # --- download dataset ---
            if (not Path(dataset_dir, 'VOCtrainval_11-May-2012.tar').exists()):
                logging.info('downloading VOCtrainval_11-May-2012.tar')
                url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
                save_file = download_file(url, dataset_dir)
                safe_extract_tar(save_file, dataset_dir)
            else:
                logging.info('VOCtrainval_11-May-2012.tar is exists (Skip Download)')
            
        # --- load annotations(instances) ---
        class_map = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'pottedplant': 15,
            'sheep': 16,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19,
        }
        
        annotation_dir = Path(dataset_dir, 'VOCdevkit', 'VOC2012', 'Annotations')
        image_dir = Path(dataset_dir, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        train_txt = open(Path(dataset_dir, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'train.txt'), 'r').read().splitlines()
        validation_split = np.clip(validation_split, 0, 0.5)
        
        n_validation = int(len(train_txt) * validation_split)
        n_train = len(train_txt) - n_validation
        
        train_tfrecord_path = str(Path(dataset_dir, 'train.tfrecord'))
        writer = tf.io.TFRecordWriter(train_tfrecord_path)
        info_json_data = []
        train_images_dir = Path(dataset_dir, 'train', 'images')
        os.makedirs(train_images_dir, exist_ok=True)
        for name in train_txt[0:n_train]:
            annotation_xml = lxml_etree.fromstring(open(Path(annotation_dir, f'{name}.xml'), 'r').read())
            annotation = parse_xml(annotation_xml, multi_tag=['object'])['annotation']
            tf_example, info_target = build_tf_example(annotation, class_map, imagefile_dir=image_dir)
            writer.write(tf_example.SerializeToString())

            # --- copy image file ---
            shutil.copy(Path(image_dir, f'{name}.jpg'), Path(train_images_dir, f'{name}.jpg'))

            # --- create info.json ---
            info_json_data.append({
                'id': name,
                'img_file': f'images/{name}.jpg',
                'target': info_target,
            })
        writer.close()
        with open(Path(dataset_dir, 'train', 'info.json'), 'w') as f:
            json.dump(info_json_data, f, ensure_ascii=False, indent=4)
        
        validation_tfrecord_path = str(Path(dataset_dir, 'validation.tfrecord'))
        writer = tf.io.TFRecordWriter(validation_tfrecord_path)
        info_json_data = []
        validation_images_dir = Path(dataset_dir, 'validation', 'images')
        os.makedirs(validation_images_dir, exist_ok=True)
        for name in train_txt[n_train::]:
            annotation_xml = lxml_etree.fromstring(open(Path(annotation_dir, f'{name}.xml'), 'r').read())
            annotation = parse_xml(annotation_xml, multi_tag=['object'])['annotation']
            tf_example, info_target = build_tf_example(annotation, class_map, imagefile_dir=image_dir)
            writer.write(tf_example.SerializeToString())

            # --- copy image file ---
            shutil.copy(Path(image_dir, f'{name}.jpg'), Path(validation_images_dir, f'{name}.jpg'))

            # --- create info.json ---
            info_json_data.append({
                'id': name,
                'img_file': f'images/{name}.jpg',
                'target': info_target,
            })
        writer.close()
        with open(Path(dataset_dir, 'validation', 'info.json'), 'w') as f:
            json.dump(info_json_data, f, ensure_ascii=False, indent=4)
        
        test_txt = open(Path(dataset_dir, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'val.txt'), 'r').read().splitlines()
        test_tfrecord_path = str(Path(dataset_dir, 'test.tfrecord'))
        writer = tf.io.TFRecordWriter(test_tfrecord_path)
        info_json_data = []
        test_images_dir = Path(dataset_dir, 'test', 'images')
        os.makedirs(test_images_dir, exist_ok=True)
        for name in test_txt:
            annotation_xml = lxml_etree.fromstring(open(Path(annotation_dir, f'{name}.xml'), 'r').read())
            annotation = parse_xml(annotation_xml, multi_tag=['object'])['annotation']
            tf_example, info_target = build_tf_example(annotation, class_map, imagefile_dir=image_dir)
            writer.write(tf_example.SerializeToString())

            # --- copy image file ---
            shutil.copy(Path(image_dir, f'{name}.jpg'), Path(test_images_dir, f'{name}.jpg'))

            # --- create info.json ---
            info_json_data.append({
                'id': name,
                'img_file': f'images/{name}.jpg',
                'target': info_target,
            })
        writer.close()
        with open(Path(dataset_dir, 'test', 'info.json'), 'w') as f:
            json.dump(info_json_data, f, ensure_ascii=False, indent=4)
        
        # --- save class name ---
        class_name_file_path = str(Path(dataset_dir, 'voc2012.names'))
        with open(class_name_file_path, 'w') as f:
            for class_name in class_map:
                f.write(f'{class_name}\n')
        
        # --- save dataset ---
        #  * MapDataset object is cannot saved to pickle file.
        #self.train_dataset = load_dataset_from_tfrecord(train_tfrecord_path, class_name_file_path, model_input_size)
        #self.validation_dataset = load_dataset_from_tfrecord(validation_tfrecord_path, class_name_file_path, model_input_size)
        #self.test_dataset = load_dataset_from_tfrecord(test_tfrecord_path, class_name_file_path, model_input_size)
        self.train_dataset = {
            'tfrecord_path': train_tfrecord_path,
            'class_name_file_path': class_name_file_path,
            'model_input_size': model_input_size,
        }
        self.validation_dataset = {
            'tfrecord_path': validation_tfrecord_path,
            'class_name_file_path': class_name_file_path,
            'model_input_size': model_input_size,
        }
        self.test_dataset = {
            'tfrecord_path': test_tfrecord_path,
            'class_name_file_path': class_name_file_path,
            'model_input_size': model_input_size,
        }
        
        # --- save output dimension ---
        self.output_dims = len(class_map)
        
        # --- set task to image classification ---
        self.dataset_type = 'img_det'
        
        # --- T.B.D ---
        self.train_x = None
        self.train_y = None
        self.validation_x = None
        self.validation_y = None
        self.test_x = None
        self.test_y = None
        
        return
        
class DataLoaderCaliforniaHousing(DataLoader):
    """DataLoaderCaliforniaHousing
    
    DataLoader class for California Housing dataset
    """
    
    def __init__(self, dataset_dir, validation_size=0.2, test_size=0.3):
        """Constructor
        
        Constructor
        
        Args:
            dataset_dir (string): dataset directory
            validation_split (float): ratio of validation data
            test_size (float): ratio of test data
        """
        from sklearn.datasets import fetch_california_housing
        
        # --- initialize super class ---
        super().__init__()
        
        self.one_hot = False
        self.output_dims = 1
        self.verified = True
        self.dataset_type = 'table_reg'
        
        california_housing = fetch_california_housing(data_home=dataset_dir)
        self.train_x = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
        self.train_y = pd.DataFrame(california_housing.target, columns=['TARGET'])
        
        self.train_valid_test_split(validation_size=validation_size, test_size=test_size)
        
        return

class DataLoaderCustom(DataLoader):
    """DataLoaderCustom
    
    Data loader for custom dataset(user dataset).
    
    Attributes:
        train_x (numpy.ndarray): Train images
        train_y (numpy.ndarray): Train labels
        validation_x (numpy.ndarray): Validation images
        validation_y (numpy.ndarray): Validation labels
        test_x (numpy.ndarray): Test images
        test_y (numpy.ndarray): Test labels
        one_hot (bool): one hot or not about labels style
        output_dims (int): output layer dimensions
        verified (bool): verified status (True: OK, False: NG)
    
    """
    
    def __init__(self):
        """Constructor
        
        Constructor
        """
        
        # --- initialize super class ---
        super().__init__()
        
        self.one_hot = True
        self.output_dims = 1
        
        return
    
    def load_data(self, meta_dir, train_dir, validation_dir=None, test_dir=None, validation_split=0.0, flatten=False, one_hot=False):
        """load_data
        
        Load data and set to class variables
        
        Args:
            train_dir (PosixPath): training data directory
            validation_dir (PosixPath): validation data directory
            test_dir (PosixPath): test data direcotry
            validation_split (float): ratio of validation data
            flatten (bool): [T.B.D] If input shape is vector([N, :]), set to True
            one_hot (bool): If the ground truth is onehot, set to True
        
        Returns:
            None
        """
        
        def _load_image_data(data_dir, key_name='img_file'):
            """_load_image_data
            
            Load custom dataset and returns the image and label
            
            Args:
                data_dir (PosixPath): dataset directory
                key_name (string): key name of meta
            """
            
            # --- Load json data ---
            json_file = Path(data_dir, 'info.json')
            df_data = pd.read_json(json_file, orient='records')
            
            # --- Get channel ---
            img = Image.open(Path(data_dir, df_data[key_name][0]))
            if (len(img.size) == 2):
                # --- Grayscale ---
                img_channel = 1
            else:
                # --- RGB ---
                img_channle = 3
            
            # --- Load images ---
            #   * Loading images should be use Pillow, because OpenCV cannot load the grayscale image as grayscale
            #     ex) Image.open('grayscale.png').size = [H, W]
            #         cv2.imread('grayscale.png').shape = [H, W, C] (Auto convert from grayscale to color)
            images = []
            labels = []
            if (img_channel == 1):
                for index, data_ in df_data.iterrows():
                    images.append(np.array(Image.open(Path(data_dir, data_[key_name])))[:, :, np.newaxis].tolist())
                    labels.append(data_['target'])
            else:
                for index, data_ in df_data.iterrows():
                    images.append(np.array(Image.open(Path(data_dir, data_[key_name]))).tolist())
                    labels.append(data_['target'])
            
            return np.array(images), np.array(labels)
        
        def _load_table_data(data_dir, feature_names):
            """ _load_table_data
            
            This function loads table data from ``info.json``
            
            Args:
                data_dir (string): data directory ``info.json`` in.
                feature_names (list of string): feature names
            
            """
            # --- Load json data ---
            json_file = Path(data_dir, 'info.json')
            df_data = pd.read_json(json_file, orient='records')
            
            # --- Split input feature and target
            X_data = df_data[feature_names]
            if ("target" in df_data.columns):
                y_data = df_data["target"]
            else:
                y_data = None
            
            return X_data, y_data
            
        # --- set parameters from arguments ---
        self.one_hot = one_hot
        
        # --- load meta data ---
        df_meta = pd.read_json(Path(meta_dir, 'info.json'), typ='series')
        if ((df_meta['task'] == 'classification') and (df_meta['input_type'] == 'image_data')):
            self.dataset_type = 'img_clf'
        elif ((df_meta['task'] == 'regression') and (df_meta['input_type'] == 'image_data')):
            self.dataset_type = 'img_reg'
        elif ((df_meta['task'] == 'classification') and (df_meta['input_type'] == 'table_data')):
            self.dataset_type = 'table_clf'
        elif ((df_meta['task'] == 'regression') and (df_meta['input_type'] == 'table_data')):
            self.dataset_type = 'table_reg'
        elif ((df_meta['task'] == 'object_detection') and (df_meta['input_type'] == 'image_data')):
            self.dataset_type = 'img_det'
        else:
            self.dataset_type = None
        
        if (df_meta['input_type'] == 'image_data'):
            for key in df_meta['keys']:
                if (key['type'] == 'image_file'):
                    key_name = key['name']
                    break
            
            # --- load training data ---
            self.train_x, self.train_y = _load_image_data(train_dir, key_name=key_name)
            
            # --- load validation data ---
            if (validation_dir is not None):
                self.validation_x, self.validation_y = _load_image_data(validation_dir, key_name=key_name)
            else:
                self.split_train_val(validation_split)
                
            # --- load test data ---
            if (test_dir is not None):
                self.test_x, self.test_y = _load_image_data(test_dir, key_name=key_name)
            
            # --- set output dims ---
            self.output_dims = len(np.unique(self.train_y))
            
        elif (df_meta['input_type'] == 'table_data'):
            # --- load input feature names ---
            feature_names = []
            for key in df_meta['keys']:
                feature_names.append(key['name'])
            
            # --- load training data ---
            self.train_x, self.train_y = _load_table_data(train_dir, feature_names)
            
            # --- load validation data ---
            if (validation_dir is not None):
                self.validation_x, self.validation_y = _load_table_data(validation_dir, feature_names)
            
            # --- load test data ---
            if (test_dir is not None):
                self.test_x, self.test_y = _load_table_data(test_dir, feature_names)
            
        return
    
    def verify(self, meta_dir, train_dir, validation_dir=None, test_dir=None):
        """verify
        
        Verify dataset
        
        Args:
            meta_dir (PosixPath): Meta data (zip extracted)
            train_dir (PosixPath): Train data (zip extracted)
            validation_dir (PosixPath): Validation data (zip extracted)
            test_dir (PosixPath): Test data (zip extracted)
        
        Returns:
            bool: Result of verification (True: OK, False: NG)
        
        """
        
        self.verified = False
        
        if (not Path(meta_dir, 'info.json').exists()):
            return False
            
        if (not Path(train_dir, 'info.json').exists()):
            return False
        
        if (validation_dir is not None):
            if (not Path(validation_dir, 'info.json').exists()):
                return False
                
        if (test_dir is not None):
            if (not Path(test_dir, 'info.json').exists()):
                return False
        
        self.verified = True
        return True

