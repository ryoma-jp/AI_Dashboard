#! -*- coding: utf-8 -*-

#---------------------------------
# Import modules
#---------------------------------
import io
import os
import fcntl
import gc
import logging
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error

from machine_learning.lib.trainer.tf_models.yolov3 import models as yolov3_models
from machine_learning.lib.trainer.tf_models.yolov3.utils import freeze_all as yolov3_freeze_all
from machine_learning.lib.data_loader.data_loader import load_dataset_from_tfrecord

#---------------------------------
# Set environ
#---------------------------------
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"

#---------------------------------
# Classes
#---------------------------------
class Trainer():
    """Trainer
    
    Base class for training
    
    Attributes:
        trainer_ctrl_fifo (str): FIFO path to control the model training
        model (keras.models.Model): model
        output_dir (str): output directory path to save the result of training and logs
        
    """
    class CustomCallback(keras.callbacks.Callback):
        def __init__(self, trainer_ctrl_fifo):
            super().__init__()
            self.trainer_ctrl_fifo = trainer_ctrl_fifo
            
        def on_train_batch_end(self, batch, logs=None):
            if (self.trainer_ctrl_fifo is not None):
                fd = os.open(self.trainer_ctrl_fifo, os.O_RDONLY | os.O_NONBLOCK)
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                flags &= ~os.O_NONBLOCK
                fcntl.fcntl(fd, fcntl.F_SETFL, flags)
                
                try:
                    command = os.read(fd, 128)
                    command = command.decode()[:-1]
                    while (True):
                        buf = os.read(fd, 65536)
                        if not buf:
                            break
                finally:
                    os.close(fd)
            
                if (command):
                    if (command == 'stop'):
                        print('End batch: recv command={}'.format(command))
                        self.model.stop_training = True
                    else:
                        print('End batch: recv unknown command={}'.format(command))
            
        def on_epoch_end(self, epoch, logs=None):
            keys = list(logs.keys())
            log_str = ''
            for key in keys[:-1]:
                log_str += '({} = {}), '.format(key, logs[key])
            log_str += '({} = {})'.format(keys[-1], logs[keys[-1]])
            print("End epoch {}: {}".format(epoch, log_str))
            
    def __init__(self, output_dir=None, model_file=None,
                 web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
                 initializer='glorot_uniform', optimizer='adam', loss='sparse_categorical_crossentropy',
                 dropout_rate=0.0, learning_rate=0.001,
                 dataset_type='img_clf', da_params=None,
                 batch_size=32, epochs=200):
        """Constructor
        
        Constructor
        
        Args:
            output_dir (:obj:`string`, optional): Output directory path
            model_file (:obj:`model_file`, optional): Trained model path
            web_app_ctrl_fifo (str): FIFO path to control Web app(Trainer -> Web app)
            trainer_ctrl_fifo (str): FIFO path to control Trainer(Web app -> Trainer)
            initializer (:obj:`string`, optional): Initializer
                - glorot_uniform: Xavier uniform distribution
                - he_normal: He normal distribution
                - lecun_normal: LeCun normal distribution
                - he_uniform: He uniform distribution
                - lecun_uniform: LeCun uniform distribution
            optimizer (:obj:`string`, optional): Optimizer
            loss (:obj:`string`, optional): Loss function
            dropout_rate (:obj:`string`, optional): Dropout rate
            learning_rate (:obj:`float`, optional): Learning rate
            dataset_type (:obj:`string`, optional): Dataset type
                - img_clf: Image classification
                - img_reg: Image regression
                - table_clf: Table data classification
                - table_reg: Table data regression
            da_params (:obj:`dict`, optional): DataAugmentation parameters
            batch_size (:obj:`int`, optional): mini batch size
            epochs (:obj:`int`, optional): EPOCHs
            decoded_preds (:obj:`dict`, optional): predictions
                - img_clf
                    {
                        'class_id': <class id>,
                        'class_name': <class name>,
                        'score': <score of class>
                    }
        """
        
        # --- Load parameters ---
        self.output_dir = output_dir
        self.web_app_ctrl_fifo = web_app_ctrl_fifo
        self.trainer_ctrl_fifo = trainer_ctrl_fifo
        self.initializer = initializer
        self.optimizer = optimizer
        self.loss = loss
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.dataset_type = dataset_type
        self.da_params = da_params
        self.batch_size = batch_size
        self.epochs = epochs
        self.decoded_preds = {}
        
        # --- Create output directory ---
        if (self.output_dir is not None):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # --- Create model ---
        def _load_model(model_file):
            if (model_file is not None):
                return keras.models.load_model(model_file), '', ''
            else:
                return None, '', ''
        
        self.model, self.input_tensor_name, self.output_tensor_name = _load_model(model_file)
        if (self.model is not None):
            self._compile_model(optimizer=self.optimizer, loss=self.loss, init_lr=self.learning_rate)
        
        return
    
    def _compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', init_lr=0.001):
        """Compile model
        
        Compile model
        
        Args:
            optimizer (string): optimizer
                - adam: Adam
                - sgd: SGD
                - adam_lrs: Adam with learning rate scheduler
                - sgd_lrs: SGD with learning rate scheduler
            loss (string): loss function
                - sparse_categorical_crossentropy
                - categorical_crossentropy
            init_lr (float): init learning rate
        
        """
        
        if (optimizer == 'adam'):
            opt = tf.keras.optimizers.Adam(learning_rate=init_lr)
        elif (optimizer == 'sgd'):
            opt = tf.keras.optimizers.SGD(learning_rate=init_lr)
        elif (optimizer == 'adam_lrs'):
            # --- parameters ---
            #  https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
            #    but, initial learning rate is default of Adam()
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                init_lr,
                decay_steps=1000,
                decay_rate=0.90,
                staircase=True)
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        elif (optimizer == 'sgd_lrs'):
            # --- parameters ---
            #  https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
            #    but, initial learning rate is default of Adam()
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                init_lr,
                decay_steps=1000,
                decay_rate=0.9,
                staircase=True)
            opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        elif (optimizer == 'momentum'):
            # --- parameters ---
            #  https://qiita.com/8128/items/2d441e46643f73c0ca19
            opt = tf.keras.optimizers.SGD(learning_rate=init_lr, decay=1e-4, momentum=0.9, nesterov=True)
        else:
            print('[ERROR] Unknown optimizer: {}'.format(optimizer))
            quit()
        
        if (self.dataset_type in ['img_clf', 'table_clf']):
            metrics = ['accuracy']
        elif (self.dataset_type in ['img_reg', 'table_reg']):
            metrics = ['mean_absolute_error', 'mean_squared_error']
        else:
            metrics = None
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics)
        
        return
    
    def fit(self, x_train, y_train,
            x_val=None, y_val=None, x_test=None, y_test=None,
            verbose=0):
        """fit
        
        Training model
        
        Args:
            x_train (:obj:`numpy.ndarray`, optional): Input values of training data
            y_train (:obj:`numpy.ndarray`, optional): Ground truth of training data
            x_val (:obj:`numpy.ndarray`, optional): Input values of validation data
            y_val (:obj:`numpy.ndarray`, optional): Ground truth of validation data
            x_test (:obj:`numpy.ndarray`, optional): Input values of test data
            y_test (:obj:`numpy.ndarray`, optional): Ground truth of test data
            verbose (:obj:`int`, optional): Log level
            
        
        """
        # --- Training ---
        os.makedirs(Path(self.output_dir, 'checkpoints'), exist_ok=True)
        checkpoint_path = Path(self.output_dir, 'checkpoints', 'model.ckpt')
        cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
        custom_callback = self.CustomCallback(self.trainer_ctrl_fifo)
        tensorboard_logdir = Path(self.output_dir, 'logs')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1)
        #callbacks = [cp_callback, es_callback]
        callbacks = [cp_callback, custom_callback, tensorboard_callback]
        
        if (self.dataset_type in ['img_clf', 'img_reg']):
            if (self.da_params is not None):
                # --- no tuning ---
                datagen = ImageDataGenerator(
                    rotation_range=self.da_params['rotation_range'],
                    width_shift_range=self.da_params['width_shift_range'],
                    height_shift_range=self.da_params['height_shift_range'],
                    zoom_range=self.da_params['zoom_range'],
                    channel_shift_range=self.da_params['channel_shift_range'],
                    horizontal_flip=self.da_params['horizontal_flip'])
            else:
                datagen = ImageDataGenerator()
            datagen.fit(x_train)
            
            if ((x_val is not None) and (y_val is not None)):
                history = self.model.fit(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                            steps_per_epoch=len(x_train)/self.batch_size, validation_data=(x_val, y_val),
                            epochs=self.epochs, callbacks=callbacks,
                            verbose=verbose)
            else:
                history = self.model.fit(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                            steps_per_epoch=len(x_train)/self.batch_size, validation_split=0.2,
                            epochs=self.epochs, callbacks=callbacks,
                            verbose=verbose)
        else:
            if ((x_val is not None) and (y_val is not None)):
                history = self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size,
                            steps_per_epoch=len(x_train)/self.batch_size, validation_data=(x_val, y_val),
                            epochs=self.epochs, callbacks=callbacks,
                            verbose=verbose)
            else:
                history = self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size,
                            steps_per_epoch=len(x_train)/self.batch_size, validation_split=0.2,
                            epochs=self.epochs, callbacks=callbacks,
                            verbose=verbose)
        
        # --- Evaluate training result ---
        if (self.dataset_type in ['img_clf', 'table_clf']):
            train_loss, train_acc = self.model.evaluate(x_train, y_train, verbose=2)
            print('Train Accuracy: {}'.format(train_acc))
            print('Train Loss: {}'.format(train_loss))
            metrics = {
                'Train Accuracy': f'{train_acc:.03f}',
                'Train Loss': f'{train_loss:.03f}',
            }
            if ((x_val is not None) and (y_val is not None)):
                val_loss, val_acc = self.model.evaluate(x_val, y_val, verbose=2)
                print('Validation Accuracy: {}'.format(val_acc))
                print('Validation Loss: {}'.format(val_loss))
                metrics['Validation Accuracy'] = f'{val_acc:.03f}'
                metrics['Validation Loss'] = f'{val_loss:.03f}'
            if ((x_test is not None) and (y_test is not None)):
                test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=2)
                print('Test Accuracy: {}'.format(test_acc))
                print('Test Loss: {}'.format(test_loss))
                metrics['Test Accuracy'] = f'{test_acc:.03f}'
                metrics['Test Loss'] = f'{test_loss:.03f}'
        else:
            train_pred = self.predict(x_train)
            train_mae = mean_absolute_error(y_train, train_pred)
            train_mse = mean_squared_error(y_train, train_pred)
            metrics = {
                'Train MAE': f'{train_mae:.03}',
                'Train MSE': f'{train_mse:.03}',
                'Train RMSE': f'{math.sqrt(train_mse):.03}',
            }
            if ((x_val is not None) and (y_val is not None)):
                val_pred = self.predict(x_val)
                val_mae = mean_absolute_error(y_val, val_pred)
                val_mse = mean_squared_error(y_val, val_pred)
                metrics['Validation MAE'] = f'{val_mae:.03}'
                metrics['Validation MSE'] = f'{val_mse:.03}'
                metrics['Validation RMSE'] = f'{math.sqrt(val_mse):.03}'
        
        # --- Save metrics ---
        os.makedirs(Path(self.output_dir, 'metrics'), exist_ok=True)
        with open(Path(self.output_dir, 'metrics', 'metrics.json'), 'w') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        
        # --- Notice the finish training to Web app ---
        if (self.web_app_ctrl_fifo is not None):
            with open(self.web_app_ctrl_fifo, 'w') as f:
                f.write('trainer_done\n')
        
        return
    
    def predict(self, x_test, get_feature_map=False):
        """predict
        
        Predict
        
        Args:
            x_test (numpy.ndarray): Input values of predicting data
                - Input shape is [N, :] when ``get_feature_map==False``
                - Input shape is [:] when ``get_feature_map==True``
            get_feature_map (bool): If set to True, returns the feature maps
        
        """
        
        if (get_feature_map):
            predictions = self.model.predict(np.expand_dims(x_test, axis=0))
        else:
            predictions = self.model.predict(x_test)
        
        return predictions
        
    def save_model(self):
        """save_model
        
        Save model
        
        """
        # --- Create the output directory ---
        model_dir = Path(self.output_dir, 'models')
        os.makedirs(Path(model_dir, 'checkpoint'), exist_ok=True)
        os.makedirs(Path(model_dir, 'saved_model'), exist_ok=True)
        os.makedirs(Path(model_dir, 'hdf5'), exist_ok=True)
        
        # --- checkpoint ---
        self.model.save_weights(Path(model_dir, 'checkpoint', 'model.ckpt'))
        
        # --- saved_model ---
        self.model.save(Path(model_dir, 'saved_model'))
        
        # --- hdf5 ---
        self.model.save(Path(model_dir, 'hdf5', 'model.h5'))
        
        return
    
    def release_memory(self):
        """release_memory
        
        Release CPU and GPU memories
        
        """
        
        keras.backend.clear_session()
        del self.model
        gc.collect()

        return
        
    def GetLabelIndex(self, label, onehot=True):
        """GetLabelIndex
        
        Get label index of ground truth
        
        Args:
            label (numpy.ndarray): Grouhd truth list of dataset
            onehot (:obj:`numpy.ndarray`, optional): If ``label`` is onehot, set to True
        
        """
        if (onehot):
            label = np.argmax(label, axis=1)
        n_category = max(label)+1
        
        return np.array([np.arange(len(label))[label==i] for i in range(n_category)])
    
    def GetSystemInfo():
        """GetSystemInfo
        
        Get system information
        
        """
        _system_info = device_lib.list_local_devices()
        system_info = []
        for info in _system_info:
            dict = {}
            dict['name'] = info.name
            dict['device_type'] = info.device_type
            dict['physical_device_desc'] = info.physical_device_desc
            system_info.append(dict)
        
        return system_info
    
    def get_importance(self, index=None):
        """Get Importance
        
        (dummy function)
        
        """
        
        return None
    
    
class TrainerKerasResNet(Trainer):
    """Trainer for Keras ResNet
    
    ResNet class
    """
    def __init__(self, input_shape, classes, output_dir=None, model_file=None, model_type='custom',
                 web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
                 initializer='glorot_uniform', optimizer='adam', loss='sparse_categorical_crossentropy',
                 dropout_rate=0.0, learning_rate=0.001,
                 dataset_type='img_clf', da_params=None,
                 batch_size=32, epochs=200):
        """Constructor
        
        Constructor
        
        Args:
            input_shape (:obj:`list`, mandatory): Input shape
            classes (:obj:`int`, mandatory): Number of classes
            output_dir (:obj:`string`, optional): Output directory path
            model_file (:obj:`model_file`, optional): Trained model path
            model_type (:obj:`string`, optional): Type of model('custom' or 'custom_deep')
            web_app_ctrl_fifo (str): FIFO path to control Web app(Trainer -> Web app)
            trainer_ctrl_fifo (str): FIFO path to control Trainer(Web app -> Trainer)
            initializer (:obj:`string`, optional): Initializer
                - glorot_uniform: Xavier uniform distribution
                - he_normal: He normal distribution
                - lecun_normal: LeCun normal distribution
                - he_uniform: He uniform distribution
                - lecun_uniform: LeCun uniform distribution
            initializer (:obj:`string`, optional): Initializer
            optimizer (:obj:`string`, optional): Optimizer
            loss (:obj:`string`, optional): Loss function
            dropout_rate (:obj:`string`, optional): Dropout rate
            learning_rate (:obj:`float`, optional): Learning rate
            dataset_type (:obj:`string`, optional): Dataset type
                - img_clf: Image classification
                - img_reg: Image regression
                - table_clf: Table data classification
                - table_reg: Table data regression
            da_params (:obj:`dict`, optional): DataAugmentation parameters
            batch_size (:obj:`int`, optional): mini batch size
            epochs (:obj:`int`, optional): EPOCHs
        """
        
        # --- Residual Block ---
        #  * Copy of TensorFlow offitial, because applicaiton cannot acces keras.applications.resnet.ResNet
        #      https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/applications/resnet.py#L212
        def block1(x, filters, kernel_size=3, stride=1, initializer='glorot_uniform', conv_shortcut=True, name=None):
            bn_axis = 3
            
            if conv_shortcut:
                shortcut = keras.layers.Conv2D(4 * filters, 1, strides=stride, kernel_initializer=initializer, name=name + '_0_conv')(x)
                shortcut = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
            else:
                shortcut = x

            x = keras.layers.Conv2D(filters, 1, strides=stride, kernel_initializer=initializer, name=name + '_1_conv')(x)
            x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
            x = keras.layers.Activation('relu', name=name + '_1_relu')(x)

            x = keras.layers.Conv2D(filters, kernel_size, padding='SAME', kernel_initializer=initializer, name=name + '_2_conv')(x)
            x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
            x = keras.layers.Activation('relu', name=name + '_2_relu')(x)

            x = keras.layers.Conv2D(4 * filters, 1, kernel_initializer=initializer, name=name + '_3_conv')(x)
            x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

            x = keras.layers.Add(name=name + '_add')([shortcut, x])
            x = keras.layers.Activation('relu', name=name + '_out')(x)
            return x
        
        # --- Residual Block stack ---
        #  * Copy of TensorFlow offitial, because applicaiton cannot acces keras.applications.resnet.ResNet
        #      https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/applications/resnet.py#L257
        def stack1(x, filters, blocks, stride1=2, dropout_rate=0.0, name=None):
            x = block1(x, filters, stride=stride1, name=name + '_block1')
            x = keras.layers.Dropout(dropout_rate)(x)
            for i in range(2, blocks + 1):
                x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
                x = keras.layers.Dropout(dropout_rate)(x)
            return x
        
        def _load_model(input_shape, classes, stack_fn, initializer='glorot_uniform', dropout_rate=0.0):
            """Load model
            
            Load model
            Customize the model structure to use ``stack_fn()``
            
            Args:
                input_shape (list): Shape of input data
                classes (int): number of class
                stack_fn: function of stacks
                initializer (string): initializer
                dropout_rate (float): Dropout rate
            """
            input = keras.layers.Input(shape=input_shape)
            bn_axis = 3
            
            x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input)
            x = keras.layers.Conv2D(64, 7, strides=2, use_bias=True, kernel_initializer=initializer, name='conv1_conv')(x)

            x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
            x = keras.layers.Activation('relu', name='conv1_relu')(x)

            x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
            x = keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
            x = keras.layers.Dropout(dropout_rate)(x)

            x = stack_fn(x, dropout_rate=dropout_rate)

            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = keras.layers.Dropout(dropout_rate)(x)
            y = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
            
            model = keras.models.Model(input, y)
            model.summary()
            
            return model, input.name, y.name
            
        def _load_model_deep(input_shape, classes, stack_fn, initializer='glorot_uniform', dropout_rate=0.0):
            """Load model
            
            Load model
            Customize the model structure to use ``stack_fn()``
            
            Args:
                - input_shape (list): Shape of input data
                - classes (int): number of class
                - stack_fn: function of stacks
                - initializer (string): initializer
                - dropout_rate (float): Dropout rate
            """
            input = keras.layers.Input(shape=input_shape)
            bn_axis = 3
            
            x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input)
            x = keras.layers.Conv2D(64, 3, strides=2, use_bias=True, kernel_initializer=initializer, name='conv1_conv')(x)

            x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
            x = keras.layers.Activation('relu', name='conv1_relu')(x)

            x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
            x = keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
            x = keras.layers.Dropout(dropout_rate)(x)

            x = stack_fn(x, dropout_rate=dropout_rate)

            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = keras.layers.Dropout(dropout_rate)(x)
            y = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
            
            model = keras.models.Model(input, y)
            model.summary()
            
            return model, input.name, y.name
            
        def _load_model_resnet50(input_shape, classes, initializer='glorot_uniform', dropout_rate=0.0, pretrained=True):
            # --- ResNet50 model of from TensorFlow ---
            #  https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
            #    dbg_mode=0: original ResNet50, dbg_mode=11: custom ResNet50
            if (pretrained):
                print('[INFO] Load ResNet50 model from keras.applications')
                model = keras.applications.resnet50.ResNet50()
            else:
                def stack_fn(x, dropout_rate=0.0):
                    x = stack1(x, 64, 3, stride1=1, dropout_rate=dropout_rate, name='conv2')
                    x = stack1(x, 128, 4, dropout_rate=dropout_rate, name='conv3')
                    x = stack1(x, 256, 6, dropout_rate=dropout_rate, name='conv4')
                    return stack1(x, 512, 3, dropout_rate=dropout_rate, name='conv5')
                
                print('[INFO] Load ResNet50 model (custom implementation)')
                model = _load_model(input_shape, classes, stack_fn, initializer=initializer, dropout_rate=dropout_rate)
                
            return model
        
        # --- Initialize base class ---
        super().__init__(output_dir=output_dir, model_file=model_file,
                         web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo,
                         initializer=initializer, optimizer=optimizer, loss=loss,
                         dropout_rate=dropout_rate, learning_rate=learning_rate,
                         dataset_type=dataset_type, da_params=da_params,
                         batch_size=batch_size, epochs=epochs)
        
        # --- Create model ---
        if (self.model is None):
            if (model_type == 'custom'):
                def stack_fn(x, dropout_rate=0.0):
                    x = stack1(x, 32, 3, stride1=1, dropout_rate=dropout_rate, name='conv2')
                    return stack1(x, 64, 4, dropout_rate=dropout_rate, name='conv3')
                
                self.model, self.input_tensor_name, self.output_tensor_name = _load_model(input_shape, classes, stack_fn, initializer=self.initializer, dropout_rate=self.dropout_rate)
                self._compile_model(optimizer=self.optimizer, loss=self.loss, init_lr=self.learning_rate)
            elif (model_type == 'custom_deep'):
                def stack_fn(x, dropout_rate=0.0):
                    x = stack1(x, 16, 18, stride1=1, dropout_rate=dropout_rate, name='conv2')
                    x = stack1(x, 32, 18, dropout_rate=dropout_rate, name='conv3')
                    return stack1(x, 64, 18, dropout_rate=dropout_rate, name='conv4')
                
                self.model, self.input_tensor_name, self.output_tensor_name = _load_model_deep(input_shape, classes, stack_fn, initializer=self.initializer, dropout_rate=self.dropout_rate)
                self._compile_model(optimizer=self.optimizer, loss=self.loss, init_lr=self.learning_rate)
            elif (model_type == 'resnet50'):
                self.model, self.input_tensor_name, self.output_tensor_name = _load_model_resnet50(input_shape, classes, initializer=self.initializer, dropout_rate=self.dropout_rate, pretrained=False)
                self._compile_model(optimizer=self.optimizer, loss=self.loss, init_lr=self.learning_rate)
            else:
                print('[ERROR] Unknown model_type: {}'.format(model_type))
                return
            
        if (self.output_dir is not None):
            keras.utils.plot_model(self.model, Path(self.output_dir, 'plot_model.png'), show_shapes=True)
        
        return
    
class TrainerKerasCNN(Trainer):
    """Trainer for Keras CNN model
    
    CNN model class
    """
    def __init__(self, input_shape, classes=10, output_dir=None, model_file=None, model_type='baseline',
                 web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
                 initializer='glorot_uniform', optimizer='adam', loss='sparse_categorical_crossentropy',
                 dropout_rate=0.0, learning_rate=0.001,
                 dataset_type='img_clf', da_params=None,
                 batch_size=32, epochs=200):
        """Constructor
        
        Constructor
        
        Args:
            input_shape (:obj:`list`, mandatory): Input shape
            classes (:obj:`int`, mandatory): Number of classes
            output_dir (:obj:`string`, optional): Output directory path
            model_file (:obj:`model_file`, optional): Trained model path
            model_type (:obj:`string`, optional): Type of model('baseline' or 'deep_model')
            web_app_ctrl_fifo (str): FIFO path to control Web app(Trainer -> Web app)
            trainer_ctrl_fifo (str): FIFO path to control Trainer(Web app -> Trainer)
            initializer (:obj:`string`, optional): Initializer
                - glorot_uniform: Xavier uniform distribution
                - he_normal: He normal distribution
                - lecun_normal: LeCun normal distribution
                - he_uniform: He uniform distribution
                - lecun_uniform: LeCun uniform distribution
            initializer (:obj:`string`, optional): Initializer
            optimizer (:obj:`string`, optional): Optimizer
            loss (:obj:`string`, optional): Loss function
            dropout_rate (:obj:`string`, optional): Dropout rate
            learning_rate (:obj:`float`, optional): Learning rate
            dataset_type (:obj:`string`, optional): Dataset type
                - img_clf: Image classification
                - img_reg: Image regression
                - table_clf: Table data classification
                - table_reg: Table data regression
            da_params (:obj:`dict`, optional): DataAugmentation parameters
            batch_size (:obj:`int`, optional): mini batch size
            epochs (:obj:`int`, optional): EPOCHs
        """
        
        def _load_model(input_shape, initializer='glorot_uniform'):
            """Load model for baseline
            
            Load model for baseline
            
            Args:
                input_shape (list): Shape of input data
                initializer (string): initializer
            """
            input = keras.layers.Input(shape=input_shape)
            
            x = keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_initializer=initializer)(input)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            x = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer)(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            x = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer)(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(64, activation='relu')(x)
            y = keras.layers.Dense(classes, activation='softmax')(x)
            
            model = keras.models.Model(input, y)
            model.summary()
            
            return model, input.name, y.name
        
        # --- モデル構築(deep_model) ---
        def _load_model_deep(input_shape, initializer='glorot_uniform'):
            """Load model for deep_model
            
            Load model for deep_model
            
            Args:
                input_shape (list): Shape of input data
                initializer (string): initializer
            """
            input = keras.layers.Input(shape=input_shape)

            x = keras.layers.Conv2D(64, (3,3), padding="SAME", activation="relu")(input)
            x = keras.layers.Conv2D(64, (3,3), padding="SAME", activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Conv2D(64, (3,3), padding="SAME", activation= "relu")(x)
            x = keras.layers.MaxPooling2D()(x)
            x = keras.layers.Dropout(0.25)(x)

            x = keras.layers.Conv2D(128, (3,3), padding="SAME", activation="relu")(x)
            x = keras.layers.Conv2D(128, (3,3), padding="SAME", activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Conv2D(128, (3,3), padding="SAME", activation="relu")(x)
            x = keras.layers.MaxPooling2D()(x)
            x = keras.layers.Dropout(0.25)(x)

            x = keras.layers.Conv2D(256, (3,3), padding="SAME", activation="relu")(x)
            x = keras.layers.Conv2D(256, (3,3), padding="SAME", activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Conv2D(256, (3,3), padding="SAME", activation="relu")(x)
            x = keras.layers.Conv2D(256, (3,3), padding="SAME", activation="relu")(x)
            x = keras.layers.Conv2D(256, (3,3), padding="SAME", activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Conv2D(512, (3,3), padding="SAME", activation="relu")(x)
            x = keras.layers.Conv2D(512, (3,3), padding="SAME", activation="relu")(x)
            x = keras.layers.GlobalAveragePooling2D()(x)

            x = keras.layers.Dense(1024, activation="relu")(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Dense(1024, activation="relu")(x)
            x = keras.layers.Dropout(0.5)(x)
            y = keras.layers.Dense(classes, activation="softmax")(x)
            
            model = keras.models.Model(input, y)
            model.summary()
            
            return model, input.name, y.name
        
        # --- Initialize base class ---
        super().__init__(output_dir=output_dir, model_file=model_file,
                         web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo,
                         initializer=initializer, optimizer=optimizer, loss=loss,
                         dropout_rate=dropout_rate, learning_rate=learning_rate,
                         dataset_type=dataset_type, da_params=da_params,
                         batch_size=batch_size, epochs=epochs)
        
        # --- Create model ---
        if (self.model is None):
            if (model_type == 'baseline'):
                self.model, self.input_tensor_name, self.output_tensor_name = _load_model(input_shape, initializer=self.initializer)
            elif (model_type == 'deep_model'):
                self.model, self.input_tensor_name, self.output_tensor_name = _load_model_deep(input_shape, initializer=self.initializer)
            else:
                print('[ERROR] Unknown model_type: {}'.format(model_type))
                quit()
        
        self._compile_model(optimizer=self.optimizer, loss=loss, init_lr=self.learning_rate)
        if (self.output_dir is not None):
            keras.utils.plot_model(self.model, Path(self.output_dir, 'plot_model.png'), show_shapes=True)
        
        return
    

class TrainerKerasMLP(Trainer):
    """Trainer for MLP model
    
    MLP model class
    """
    def __init__(self, input_shape, classes=10, output_dir=None, model_file=None,
                 web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
                 initializer='glorot_uniform', optimizer='adam', loss='sparse_categorical_crossentropy',
                 dropout_rate=0.0, learning_rate=0.001,
                 dataset_type='img_clf', da_params=None,
                 batch_size=32, epochs=200,
                 num_of_hidden_nodes='128,64'):
        """Constructor
        
        Constructor
        
        Args:
            input_shape (:obj:`list`, mandatory): Input shape
            classes (:obj:`int`, mandatory): Number of classes
            output_dir (:obj:`string`, optional): Output directory path
            model_file (:obj:`model_file`, optional): Trained model path
            web_app_ctrl_fifo (str): FIFO path to control Web app(Trainer -> Web app)
            trainer_ctrl_fifo (str): FIFO path to control Trainer(Web app -> Trainer)
            initializer (:obj:`string`, optional): Initializer
                - glorot_uniform: Xavier uniform distribution
                - he_normal: He normal distribution
                - lecun_normal: LeCun normal distribution
                - he_uniform: He uniform distribution
                - lecun_uniform: LeCun uniform distribution
            initializer (:obj:`string`, optional): Initializer
            optimizer (:obj:`string`, optional): Optimizer
            loss (:obj:`string`, optional): Loss function
            dropout_rate (:obj:`string`, optional): Dropout rate
            learning_rate (:obj:`float`, optional): Learning rate
            dataset_type (:obj:`string`, optional): Dataset type
                - img_clf: Image classification
                - img_reg: Image regression
                - table_clf: Table data classification
                - table_reg: Table data regression
            da_params (:obj:`dict`, optional): DataAugmentation parameters
            batch_size (:obj:`int`, optional): mini batch size
            epochs (:obj:`int`, optional): EPOCHs
        """
        
        def _load_model(input_shape):
            """Load model
            
            Load model
            
            Args:
                input_shape (list): Shape of input data
            """
            if (self.dataset_type in ['img_clf', 'table_clf']):
                hidden_activation = 'relu'
                output_activation = 'softmax'
            else:
                hidden_activation = 'sigmoid'
                output_activation = 'linear'
            
            input = keras.layers.Input(shape=input_shape)
            x = keras.layers.Flatten()(input)
            for i, num in enumerate(self.num_of_hidden_nodes):
                x = keras.layers.Dense(num,
                                       kernel_initializer=self.initializer,
                                       bias_initializer='zeros',
                                       activation=hidden_activation,
                                       name=f'dense_{i}')(x)
            y = keras.layers.Dense(classes, 
                                   kernel_initializer=self.initializer,
                                   bias_initializer='zeros',
                                   activation=output_activation)(x)
            
            model = keras.models.Model(input, y)
            model.summary()
            
            return model, input.name, y.name
        
        # --- Get MLP parameter ---
        self.num_of_hidden_nodes = pd.read_csv(io.StringIO(num_of_hidden_nodes), header=None, skipinitialspace=True).values[0].tolist()
        
        # --- Initialize base class ---
        super().__init__(output_dir=output_dir, model_file=model_file,
                         web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo,
                         initializer=initializer, optimizer=optimizer, loss=loss,
                         dropout_rate=dropout_rate, learning_rate=learning_rate,
                         dataset_type=dataset_type, da_params=da_params,
                         batch_size=batch_size, epochs=epochs)
        
        # --- Create model ---
        if (self.model is None):
            self.model, self.input_tensor_name, self.output_tensor_name = _load_model(input_shape)
            self._compile_model(optimizer=self.optimizer, loss=self.loss, init_lr=self.learning_rate)
            if (self.output_dir is not None):
                keras.utils.plot_model(self.model, Path(self.output_dir, 'plot_model.png'), show_shapes=True)
        
        return
    
class TrainerKerasYOLOv3(Trainer):
    """Trainer for Object Detection model
    
    Object Detection model class
    """
    def __init__(self, input_shape, classes=80, output_dir=None, model_file=None, model_type='YOLOv3',
                 web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
                 initializer='glorot_uniform', optimizer='adam',
                 dropout_rate=0.0, learning_rate=0.001,
                 dataset_type='img_clf', da_params=None,
                 batch_size=32, epochs=200):
        """Constructor
        
        Constructor
        
        Args:
            input_shape (:obj:`list`, mandatory): Input shape
            classes (:obj:`int`, mandatory): Number of classes
            output_dir (:obj:`string`, optional): Output directory path
            model_file (:obj:`model_file`, optional): Trained model path
            model_type (:obj:`string`, optional): Type of model('YOLOv3' or YOLOv3-Tiny)
            web_app_ctrl_fifo (str): FIFO path to control Web app(Trainer -> Web app)
            trainer_ctrl_fifo (str): FIFO path to control Trainer(Web app -> Trainer)
            initializer (:obj:`string`, optional): Initializer
                - glorot_uniform: Xavier uniform distribution
                - he_normal: He normal distribution
                - lecun_normal: LeCun normal distribution
                - he_uniform: He uniform distribution
                - lecun_uniform: LeCun uniform distribution
            initializer (:obj:`string`, optional): Initializer
            optimizer (:obj:`string`, optional): Optimizer
            dropout_rate (:obj:`string`, optional): Dropout rate
            learning_rate (:obj:`float`, optional): Learning rate
            dataset_type (:obj:`string`, optional): Dataset type
                - img_clf: Image classification
                - img_reg: Image regression
                - table_clf: Table data classification
                - table_reg: Table data regression
            da_params (:obj:`dict`, optional): DataAugmentation parameters
            batch_size (:obj:`int`, optional): mini batch size
            epochs (:obj:`int`, optional): EPOCHs
        """
        
        def _load_model(input_shape, classes):
            """Load model for simple model
            
            Load model for simple model
            
            Args:
                input_shape (list): Shape of input data
                classes (int): Number of class
            """
            model = yolov3_models.YoloV3(size=input_shape[0], classes=classes, training=True)
            model.summary()
            
            return model, None, None
        
        # --- Initialize base class ---
        super().__init__(output_dir=output_dir, model_file=model_file,
                         web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo,
                         initializer=initializer, optimizer=optimizer, loss=None,
                         dropout_rate=dropout_rate, learning_rate=learning_rate,
                         dataset_type=dataset_type, da_params=da_params,
                         batch_size=batch_size, epochs=epochs)
        
        self.anchors = yolov3_models.yolo_anchors
        self.anchor_masks = yolov3_models.yolo_anchor_masks
        print(self.anchors)
        print(classes)
        self.loss = [yolov3_models.YoloLoss(self.anchors[mask], classes=classes) for mask in self.anchor_masks]
        print(self.loss)
        
        # --- Create model ---
        if (self.model is None):
            self.model, self.input_tensor_name, self.output_tensor_name = _load_model(input_shape, classes)
            yolov3_freeze_all(self.model.get_layer('yolo_darknet'))
            
            self._compile_model(optimizer=self.optimizer, loss=self.loss, init_lr=self.learning_rate)
            if (self.output_dir is not None):
                keras.utils.plot_model(self.model, Path(self.output_dir, 'plot_model.png'), show_shapes=True)
        
        return
    
    def fit(self, dict_train_dataset, dict_val_dataset):
        @tf.function
        def transform_targets_for_output(y_true, grid_size, anchor_idxs):
            # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
            N = tf.shape(y_true)[0]

            # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
            y_true_out = tf.zeros(
                (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

            anchor_idxs = tf.cast(anchor_idxs, tf.int32)

            indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
            updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
            idx = 0
            for i in tf.range(N):
                for j in tf.range(tf.shape(y_true)[1]):
                    if tf.equal(y_true[i][j][2], 0):
                        continue
                    anchor_eq = tf.equal(
                        anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

                    if tf.reduce_any(anchor_eq):
                        box = y_true[i][j][0:4]
                        box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                        anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                        grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                        # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                        indexes = indexes.write(
                            idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                        updates = updates.write(
                            idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                        idx += 1

            # tf.print(indexes.stack())
            # tf.print(updates.stack())

            return tf.tensor_scatter_nd_update(
                y_true_out, indexes.stack(), updates.stack())


        def transform_targets(y_train, anchors, anchor_masks, size):
            y_outs = []
            grid_size = size // 32

            # calculate anchor index for true boxes
            anchors = tf.cast(anchors, tf.float32)
            anchor_area = anchors[..., 0] * anchors[..., 1]
            box_wh = y_train[..., 2:4] - y_train[..., 0:2]
            box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                             (1, 1, tf.shape(anchors)[0], 1))
            box_area = box_wh[..., 0] * box_wh[..., 1]
            intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
                tf.minimum(box_wh[..., 1], anchors[..., 1])
            iou = intersection / (box_area + anchor_area - intersection)
            anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
            anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

            y_train = tf.concat([y_train, anchor_idx], axis=-1)

            for anchor_idxs in anchor_masks:
                y_outs.append(transform_targets_for_output(
                    y_train, grid_size, anchor_idxs))
                grid_size *= 2

            return tuple(y_outs)
        
        def transform_images(x_train, size):
            x_train = tf.image.resize(x_train, (size, size))
            x_train = x_train / 255
            return x_train
        
        # --- prepare the training dataset ---
        print(dict_train_dataset)
        print(dict_val_dataset)
        train_dataset = load_dataset_from_tfrecord(
            dict_train_dataset['tfrecord_path'], 
            dict_train_dataset['class_name_file_path'],
            dict_train_dataset['model_input_size'])
        train_dataset = train_dataset.shuffle(buffer_size=512)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.map(lambda x, y: (
            transform_images(x, dict_train_dataset['model_input_size']),
            transform_targets(y, self.anchors, self.anchor_masks, dict_train_dataset['model_input_size'])))
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        
        # --- prepare the validation dataset ---
        val_dataset = load_dataset_from_tfrecord(
            dict_val_dataset['tfrecord_path'], 
            dict_val_dataset['class_name_file_path'],
            dict_val_dataset['model_input_size'])
        val_dataset = val_dataset.batch(self.batch_size)
        val_dataset = val_dataset.map(lambda x, y: (
            transform_images(x, dict_val_dataset['model_input_size']),
            transform_targets(y, self.anchors, self.anchor_masks, dict_val_dataset['model_input_size'])))
        
        print(train_dataset)
        print(val_dataset)
        
        
        os.makedirs(Path(self.output_dir, 'checkpoints'), exist_ok=True)
        checkpoint_path = Path(self.output_dir, 'checkpoints', 'model.ckpt')
        cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
        custom_callback = self.CustomCallback(self.trainer_ctrl_fifo)
        tensorboard_logdir = Path(self.output_dir, 'logs')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1)
        #callbacks = [cp_callback, es_callback]
        callbacks = [cp_callback, custom_callback, tensorboard_callback]

        history = self.model.fit(train_dataset,
                            epochs=self.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)
    
        # --- Notice the finish training to Web app ---
        if (self.web_app_ctrl_fifo is not None):
            with open(self.web_app_ctrl_fifo, 'w') as f:
                f.write('trainer_done\n')
    
def main():
    """Main module
    
    Main module for Unit test of Keras Trainer.
    
    """
    import argparse
    def _argparse():
        parser = argparse.ArgumentParser(description='Unit test of Keras Trainer',
                    formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument('--test_mode', dest='test_mode', type=str, default='ResNet', required=False, \
                help='テストモード(ResNet)')

        args = parser.parse_args()
        return args

    # --- Arguments processing ---
    args = _argparse()
    print(args.test_mode)
    
    # --- Unit test ---
    if (args.test_mode == 'ResNet'):
        trainer = TrainerResNet([224, 224, 3], 1000, output_dir=None, model_type='resnet50')
    else:
        print('[ERROR] Unknown test_mode: {}'.format(args.test_mode))
    
    return

    
if __name__ == '__main__':
    main()
