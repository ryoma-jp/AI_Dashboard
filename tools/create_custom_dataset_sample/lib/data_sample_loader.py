"""Data Samples Loader

This library loads data samples from downloaded files
"""

import os
import urllib
import json
import pickle
import cv2
import numpy as np
import pandas as pd

def load_mnist_dataset(input_dir):
    """Load MNIST data samples

    This function loads data samples as ``numpy.ndarray``.

    Args:
        input_dir (string): specify the directory included the MNIST python version is extracted.

    Returns:
        MNIST data samples

        - train_images (numpy.ndarray): image samples for training (shape: [NHW])
        - train_labels (numpy.ndarray): target labels for training (shape: [N])
        - test_images (numpy.ndarray): image samples for test (shape: [NHW])
        - test_labels (numpy.ndarray): target labels for test (shape: [N])
    """
    
    # --- load training data ---
    f = open(os.path.join(input_dir, 'train-images-idx3-ubyte'))
    byte_data = np.fromfile(f, dtype=np.uint8)
    
    n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
    img_h = (byte_data[8] << 24) | (byte_data[9] << 16) | (byte_data[10] << 8) | (byte_data[11])
    img_w = (byte_data[12] << 24) | (byte_data[13] << 16) | (byte_data[14] << 8) | (byte_data[15])
    train_images = byte_data[16:].reshape(n_items, img_h, img_w)
    
    # --- load training label ---
    f = open(os.path.join(input_dir, 'train-labels-idx1-ubyte'))
    byte_data = np.fromfile(f, dtype=np.uint8)
    n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
    train_labels = byte_data[8:]
    
    # --- load test data ---
    f = open(os.path.join(input_dir, 't10k-images-idx3-ubyte'))
    byte_data = np.fromfile(f, dtype=np.uint8)
    
    n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
    img_h = (byte_data[8] << 24) | (byte_data[9] << 16) | (byte_data[10] << 8) | (byte_data[11])
    img_w = (byte_data[12] << 24) | (byte_data[13] << 16) | (byte_data[14] << 8) | (byte_data[15])
    test_images = byte_data[16:].reshape(n_items, img_h, img_w)
    
    # --- load test label ---
    f = open(os.path.join(input_dir, 't10k-labels-idx1-ubyte'))
    byte_data = np.fromfile(f, dtype=np.uint8)
    n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
    test_labels = byte_data[8:]
    
    return train_images, train_labels, test_images, test_labels
    

def load_cifar10_dataset(input_dir='input'):
    """Load CIFAR-10 data samples

    This function loads data samples as ``numpy.ndarray``.

    Args:
        input_dir (string): specify the directory included the CIFAR-10 python version is extracted.

    Returns:
        CIFAR-10 data samples

        - train_images (numpy.ndarray): image samples for training (shape: [NHWC], channel shape: [RGB])
        - train_labels (numpy.ndarray): target labels for training (shape: [N])
        - test_images (numpy.ndarray): image samples for test (shape: [NHWC], channel shape: [RGB])
        - test_labels (numpy.ndarray): target labels for test (shape: [N])
    """

    # --- local function for unpickle ---
    def _unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    # --- training data ---
    train_data_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    dict_data = _unpickle(os.path.join(input_dir, train_data_list[0]))
    train_images = dict_data[b'data']
    train_labels = dict_data[b'labels'].copy()
    for train_data in train_data_list[1:]:
        dict_data = _unpickle(os.path.join(input_dir, train_data))
        train_images = np.vstack((train_images, dict_data[b'data']))
        train_labels = np.hstack((train_labels, dict_data[b'labels']))
    
    # --- test data ---
    test_data = "test_batch"
    dict_data = _unpickle(os.path.join(input_dir, test_data))
    test_images = dict_data[b'data']
    test_labels = dict_data[b'labels'].copy()
    
    # --- transpose: [N, C, H, W] -> [N, H, W, C] ---
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return train_images, np.array(train_labels), test_images, np.array(test_labels)
    
