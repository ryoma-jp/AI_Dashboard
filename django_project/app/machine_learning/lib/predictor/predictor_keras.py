"""Predictor for Keras

This file describe about the prediction process using Keras
"""

import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow import keras

class Predictor():
    """Predictor
    
    This class specifies the process of loading model and predicting.
    """
    
    def __init__(self, model_name):
        """Constructor
        
        This function is the construction of predictor.
        
        Args:
            model_name(str): specify the model name
                               - ResNet50
        """
        
        if (model_name == 'ResNet50'):
            self.pretrained_model = keras.applications.ResNet50()
            self.input_shape = [224, 224, 3]
        else:
            pass
    
    def predict(self, x):
        """Predict
        
        This function predicts ``x`` using ``self.pretrained_model``
        
        Args:
            x (np.array): input data
                            - image: shape is [[N]+``self.input_shape``], channel is [R, G, B]
        """
        
        tf_x = tf.convert_to_tensor(x * (2.0 / 255.0) - 1.0, dtype=tf.float32)
        prediction = self.pretrained_model.predict(tf_x)
        
        return prediction
    

'''
#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
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

#---------------------------------
# 環境変数設定
#---------------------------------
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"

#---------------------------------
# クラス; 学習モジュール基底クラス
#---------------------------------
class Trainer():
    """Trainer
    
    学習モジュールの基底クラス
    
    Attributes:
        trainer_ctrl_fifo (str): モデル学習制御用のFIFOのパス
        model (keras.models.Model): 学習対象のモデル
        output_dir (str): 学習結果やログ出力用のディレクトリのパス
        
    """
    # --- カスタムコールバック ---
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
            
    # --- コンストラクタ ---
    def __init__(self, output_dir=None, model_file=None,
                 web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
                 initializer='glorot_uniform', optimizer='adam', loss='sparse_categorical_crossentropy',
                 dropout_rate=0.0, learning_rate=0.001,
                 dataset_type='img_clf', da_params=None,
                 batch_size=32, epochs=200):
        """Constructor
        
        コンストラクタ
        
        Args:
            output_dir (:obj:`string`, optional): 出力ディレクトリのパス
            model_file (:obj:`model_file`, optional): 学習済みモデルのパス
            web_app_ctrl_fifo (str): Webアプリ制御用FIFOのパス(TrainerがWebアプリを制御)
            trainer_ctrl_fifo (str): Trainer制御用FIFOのパス(WebアプリがTrainerを制御)
            initializer (:obj:`string`, optional): Initializer
                - glorot_uniform: Xavierの一様分布
                - he_normal: Heの正規分布
                - lecun_normal: LeCunの正規分布
                - he_uniform: Heの一様分布
                - lecun_uniform: LeCunの一様分布
            optimizer (:obj:`string`, optional): Optimizer
            loss (:obj:`string`, optional): Loss function
            dropout_rate (:obj:`string`, optional): Dropout rate
            learning_rate (:obj:`float`, optional): Learning rate
            dataset_type (:obj:`string`, optional): Dataset type
                - img_clf: 画像分類
                - img_reg: 画像回帰
                - table_clf: テーブルデータ分類
                - table_reg: テーブルデータ回帰
            da_params (:obj:`dict`, optional): DataAugmentationパラメータ
            batch_size (:obj:`int`, optional): ミニバッチ数
            epochs (:obj:`int`, optional): 学習EPOCH数
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
        
        # --- 出力ディレクトリ作成 ---
        if (self.output_dir is not None):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # --- モデル構築 ---
        def _load_model(model_file):
            if (model_file is not None):
                return keras.models.load_model(model_file)
            else:
                return None
        
        self.model = _load_model(model_file)
        if (self.model is not None):
            self._compile_model(optimizer=self.optimizer, loss=self.loss, init_lr=self.learning_rate)
        
        return
    
    # --- モデルの構成 ---
    #   * lr_decay: 学習率減衰する(=True)，しない(=False; default)を指定
    def _compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', init_lr=0.001):
        
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
        else:
            metrics = ['mean_absolute_error', 'mean_squared_error']
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics)
        
        return
    
    # --- 学習 ---
    def fit(self, x_train, y_train,
            x_val=None, y_val=None, x_test=None, y_test=None,
            verbose=0):
        """fit
        
        AIモデルの学習を実行する
        
        Args:
            x_train (:obj:`numpy.ndarray`, optional): 学習データの入力値
            y_train (:obj:`numpy.ndarray`, optional): 学習データの真値
            x_val (:obj:`numpy.ndarray`, optional): Validationデータの入力値
            y_val (:obj:`numpy.ndarray`, optional): Validationデータの真値
            x_test (:obj:`numpy.ndarray`, optional): Testデータの入力値
            y_test (:obj:`numpy.ndarray`, optional): Testデータの真値
            verbose (:obj:`int`, optional): ログの出力レベル
            
        
        """
        # --- 学習 ---
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
        
        # --- 学習結果を評価 ---
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
        
        # --- メトリクスを保存 ---
        os.makedirs(Path(self.output_dir, 'metrics'), exist_ok=True)
        with open(Path(self.output_dir, 'metrics', 'metrics.json'), 'w') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        
        # --- 学習完了をアプリへ通知 ---
        if (self.web_app_ctrl_fifo is not None):
            with open(self.web_app_ctrl_fifo, 'w') as f:
                f.write('trainer_done\n')
        
        return
    
    # --- 推論 ---
    def predict(self, x_test):
        """predict
        
        推論を実行する
        
        Args:
            x_test (numpy.ndarray): 推論対象データの入力値
        
        """
        
        predictions = self.model.predict(x_test)
        return predictions
        
    # --- モデル保存 ---
    def save_model(self):
        """save_model
        
        モデルを保存する
        
        """
        # --- 保存先ディレクトリ作成 ---
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
    
    # --- メモリリソース解放(セッションのクリア) ---
    def release_memory(self):
        """release_memory
        
        学習で使用したメモリリソースを解放する(CPU/GPU)
        
        """
        
        keras.backend.clear_session()
        del self.model
        gc.collect()

        return
        
    # --- ラベルインデックス取得 ---
    def GetLabelIndex(self, label, onehot=True):
        """GetLabelIndex
        
        真値のラベルインデックスを取得する
        
        Args:
            label (numpy.ndarray): データセットの真値リスト
            onehot (:obj:`numpy.ndarray`, optional): labelがonehotの場合にTrueを指定
        
        """
        if (onehot):
            label = np.argmax(label, axis=1)
        n_category = max(label)+1
        
        return np.array([np.arange(len(label))[label==i] for i in range(n_category)])
    
    # --- システム情報を取得 ---
    def GetSystemInfo():
        """GetSystemInfo
        
        システム情報を取得する
        
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
    
    
#---------------------------------
# クラス; ResNet学習モジュール
#---------------------------------
class TrainerKerasResNet(Trainer):
    # --- コンストラクタ ---
    def __init__(self, input_shape, classes, output_dir=None, model_file=None, model_type='custom',
                 web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
                 initializer='glorot_uniform', optimizer='adam', loss='sparse_categorical_crossentropy',
                 dropout_rate=0.0, learning_rate=0.001,
                 dataset_type='img_clf', da_params=None,
                 batch_size=32, epochs=200):
        """Constructor
        
        コンストラクタ
        
        Args:
            input_shape (:obj:`list`, mandatory): Input shape
            classes (:obj:`int`, mandatory): Number of classes
            output_dir (:obj:`string`, optional): 出力ディレクトリのパス
            model_file (:obj:`model_file`, optional): 学習済みモデルのパス
            model_type (:obj:`string`, optional): モデルの種類('custom' or 'custom_deep')
            web_app_ctrl_fifo (str): Webアプリ制御用FIFOのパス(TrainerがWebアプリを制御)
            trainer_ctrl_fifo (str): Trainer制御用FIFOのパス(WebアプリがTrainerを制御)
            initializer (:obj:`string`, optional): Initializer
                - glorot_uniform: Xavierの一様分布
                - he_normal: Heの正規分布
                - lecun_normal: LeCunの正規分布
                - he_uniform: Heの一様分布
                - lecun_uniform: LeCunの一様分布
            initializer (:obj:`string`, optional): Initializer
            optimizer (:obj:`string`, optional): Optimizer
            loss (:obj:`string`, optional): Loss function
            dropout_rate (:obj:`string`, optional): Dropout rate
            learning_rate (:obj:`float`, optional): Learning rate
            dataset_type (:obj:`string`, optional): Dataset type
                - img_clf: 画像分類
                - img_reg: 画像回帰
                - table_clf: テーブルデータ分類
                - table_reg: テーブルデータ回帰
            da_params (:obj:`dict`, optional): DataAugmentationパラメータ
            batch_size (:obj:`int`, optional): ミニバッチ数
            epochs (:obj:`int`, optional): 学習EPOCH数
        """
        
        # --- Residual Block ---
        #  * アプリケーションからkeras.applications.resnet.ResNetにアクセスできない為，
        #    必要なモジュールをTensorFlow公式からコピー
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
        #  * アプリケーションからkeras.applications.resnet.ResNetにアクセスできない為，
        #    必要なモジュールをTensorFlow公式からコピー
        #      https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/applications/resnet.py#L257
        def stack1(x, filters, blocks, stride1=2, dropout_rate=0.0, name=None):
            x = block1(x, filters, stride=stride1, name=name + '_block1')
            x = keras.layers.Dropout(dropout_rate)(x)
            for i in range(2, blocks + 1):
                x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
                x = keras.layers.Dropout(dropout_rate)(x)
            return x
        
        # --- モデル構築 ---
        #  * stack_fn()の関数ポインタを引数に設定してカスタマイズ
        def _load_model(input_shape, classes, stack_fn, initializer='glorot_uniform', dropout_rate=0.0):
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
            x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
            
            model = keras.models.Model(input, x)
            model.summary()
            
            return model
            
        # --- モデル構築 ---
        #  * stack_fn()の関数ポインタを引数に設定してカスタマイズ
        def _load_model_deep(input_shape, classes, stack_fn, initializer='glorot_uniform', dropout_rate=0.0):
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
            x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
            
            model = keras.models.Model(input, x)
            model.summary()
            
            return model
            
        def _load_model_resnet50(input_shape, classes, initializer='glorot_uniform', dropout_rate=0.0, pretrained=True):
            # --- TensorFlowのResNet50のモデル ---
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
        
        # --- 基底クラスの初期化 ---
        super().__init__(output_dir=output_dir, model_file=model_file,
                         web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo,
                         initializer=initializer, optimizer=optimizer, loss=loss,
                         dropout_rate=dropout_rate, learning_rate=learning_rate,
                         dataset_type=dataset_type, da_params=da_params,
                         batch_size=batch_size, epochs=epochs)
        
        # --- モデル構築 ---
        if (self.model is None):
            if (model_type == 'custom'):
                def stack_fn(x, dropout_rate=0.0):
                    x = stack1(x, 32, 3, stride1=1, dropout_rate=dropout_rate, name='conv2')
                    return stack1(x, 64, 4, dropout_rate=dropout_rate, name='conv3')
                
                self.model = _load_model(input_shape, classes, stack_fn, initializer=self.initializer, dropout_rate=self.dropout_rate)
                self._compile_model(optimizer=self.optimizer, loss=self.loss, init_lr=self.learning_rate)
            elif (model_type == 'custom_deep'):
                def stack_fn(x, dropout_rate=0.0):
                    x = stack1(x, 16, 18, stride1=1, dropout_rate=dropout_rate, name='conv2')
                    x = stack1(x, 32, 18, dropout_rate=dropout_rate, name='conv3')
                    return stack1(x, 64, 18, dropout_rate=dropout_rate, name='conv4')
                
                self.model = _load_model_deep(input_shape, classes, stack_fn, initializer=self.initializer, dropout_rate=self.dropout_rate)
                self._compile_model(optimizer=self.optimizer, loss=self.loss, init_lr=self.learning_rate)
            elif (model_type == 'resnet50'):
                self.model = _load_model_resnet50(input_shape, classes, initializer=self.initializer, dropout_rate=self.dropout_rate, pretrained=False)
                self._compile_model(optimizer=self.optimizer, loss=self.loss, init_lr=self.learning_rate)
            else:
                print('[ERROR] Unknown model_type: {}'.format(model_type))
                return
            
        if (self.output_dir is not None):
            keras.utils.plot_model(self.model, Path(self.output_dir, 'plot_model.png'), show_shapes=True)
        
        return
    
#---------------------------------
# クラス; CNN学習モジュール
#---------------------------------
class TrainerKerasCNN(Trainer):
    # --- コンストラクタ ---
    def __init__(self, input_shape, classes=10, output_dir=None, model_file=None, model_type='baseline',
                 web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
                 initializer='glorot_uniform', optimizer='adam', loss='sparse_categorical_crossentropy',
                 dropout_rate=0.0, learning_rate=0.001,
                 dataset_type='img_clf', da_params=None,
                 batch_size=32, epochs=200):
        """Constructor
        
        コンストラクタ
        
        Args:
            input_shape (:obj:`list`, mandatory): Input shape
            classes (:obj:`int`, mandatory): Number of classes
            output_dir (:obj:`string`, optional): 出力ディレクトリのパス
            model_file (:obj:`model_file`, optional): 学習済みモデルのパス
            model_type (:obj:`string`, optional): モデルの種類('baseline' or 'deep_model')
            web_app_ctrl_fifo (str): Webアプリ制御用FIFOのパス(TrainerがWebアプリを制御)
            trainer_ctrl_fifo (str): Trainer制御用FIFOのパス(WebアプリがTrainerを制御)
            initializer (:obj:`string`, optional): Initializer
                - glorot_uniform: Xavierの一様分布
                - he_normal: Heの正規分布
                - lecun_normal: LeCunの正規分布
                - he_uniform: Heの一様分布
                - lecun_uniform: LeCunの一様分布
            initializer (:obj:`string`, optional): Initializer
            optimizer (:obj:`string`, optional): Optimizer
            loss (:obj:`string`, optional): Loss function
            dropout_rate (:obj:`string`, optional): Dropout rate
            learning_rate (:obj:`float`, optional): Learning rate
            dataset_type (:obj:`string`, optional): Dataset type
                - img_clf: 画像分類
                - img_reg: 画像回帰
                - table_clf: テーブルデータ分類
                - table_reg: テーブルデータ回帰
            da_params (:obj:`dict`, optional): DataAugmentationパラメータ
            batch_size (:obj:`int`, optional): ミニバッチ数
            epochs (:obj:`int`, optional): 学習EPOCH数
        """
        
        # --- モデル構築(baseline) ---
        def _load_model(input_shape, initializer='glorot_uniform'):
            model = keras.models.Sequential()
            model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_initializer=initializer))
            model.add(keras.layers.MaxPooling2D((2, 2)))
            model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer))
            model.add(keras.layers.MaxPooling2D((2, 2)))
            model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer))
            model.add(keras.layers.MaxPooling2D((2, 2)))
            model.add(keras.layers.Flatten(input_shape=input_shape))
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dense(classes, activation='softmax'))
            
            model.summary()
            
            return model
        
        # --- モデル構築(deep_model) ---
        def _load_model_deep(input_shape, initializer='glorot_uniform'):
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
            x = keras.layers.Dense(classes, activation="softmax")(x)
            
            model = keras.models.Model(input, x)
            model.summary()
            
            return model
        
        # --- 基底クラスの初期化 ---
        super().__init__(output_dir=output_dir, model_file=model_file,
                         web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo,
                         initializer=initializer, optimizer=optimizer, loss=loss,
                         dropout_rate=dropout_rate, learning_rate=learning_rate,
                         dataset_type=dataset_type, da_params=da_params,
                         batch_size=batch_size, epochs=epochs)
        
        # --- モデル構築 ---
        if (self.model is None):
            if (model_type == 'baseline'):
                self.model = _load_model(input_shape, initializer=self.initializer)
            elif (model_type == 'deep_model'):
                self.model = _load_model_deep(input_shape, initializer=self.initializer)
            else:
                print('[ERROR] Unknown model_type: {}'.format(model_type))
                quit()
        
        self._compile_model(optimizer=self.optimizer, loss=loss, init_lr=self.learning_rate)
        if (self.output_dir is not None):
            keras.utils.plot_model(self.model, Path(self.output_dir, 'plot_model.png'), show_shapes=True)
        
        return
    

#---------------------------------
# クラス; MLP学習モジュール
#---------------------------------
class TrainerKerasMLP(Trainer):
    # --- コンストラクタ ---
    def __init__(self, input_shape, classes=10, output_dir=None, model_file=None,
                 web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
                 initializer='glorot_uniform', optimizer='adam', loss='sparse_categorical_crossentropy',
                 dropout_rate=0.0, learning_rate=0.001,
                 dataset_type='img_clf', da_params=None,
                 batch_size=32, epochs=200,
                 num_of_hidden_nodes='128,64'):
        """Constructor
        
        コンストラクタ
        
        Args:
            input_shape (:obj:`list`, mandatory): Input shape
            classes (:obj:`int`, mandatory): Number of classes
            output_dir (:obj:`string`, optional): 出力ディレクトリのパス
            model_file (:obj:`model_file`, optional): 学習済みモデルのパス
            web_app_ctrl_fifo (str): Webアプリ制御用FIFOのパス(TrainerがWebアプリを制御)
            trainer_ctrl_fifo (str): Trainer制御用FIFOのパス(WebアプリがTrainerを制御)
            initializer (:obj:`string`, optional): Initializer
                - glorot_uniform: Xavierの一様分布
                - he_normal: Heの正規分布
                - lecun_normal: LeCunの正規分布
                - he_uniform: Heの一様分布
                - lecun_uniform: LeCunの一様分布
            initializer (:obj:`string`, optional): Initializer
            optimizer (:obj:`string`, optional): Optimizer
            loss (:obj:`string`, optional): Loss function
            dropout_rate (:obj:`string`, optional): Dropout rate
            learning_rate (:obj:`float`, optional): Learning rate
            dataset_type (:obj:`string`, optional): Dataset type
                - img_clf: 画像分類
                - img_reg: 画像回帰
                - table_clf: テーブルデータ分類
                - table_reg: テーブルデータ回帰
            da_params (:obj:`dict`, optional): DataAugmentationパラメータ
            batch_size (:obj:`int`, optional): ミニバッチ数
            epochs (:obj:`int`, optional): 学習EPOCH数
        """
        
        # --- モデル構築 ---
        def _load_model(input_shape):
            if (self.dataset_type in ['img_clf', 'table_clf']):
                hidden_activation = 'relu'
                output_activation = 'softmax'
            else:
                hidden_activation = 'sigmoid'
                output_activation = 'linear'
            
            model = keras.models.Sequential()
            model.add(keras.layers.Flatten(input_shape=input_shape))
            for num in self.num_of_hidden_nodes:
                model.add(keras.layers.Dense(num,
                                             kernel_initializer=self.initializer,
                                             bias_initializer='zeros',
                                             activation=hidden_activation))
            model.add(keras.layers.Dense(classes,
                                         kernel_initializer=self.initializer,
                                         bias_initializer='zeros',
                                         activation=output_activation))
            
            model.summary()
            
            return model
        
        # --- MLP固有パラメータの取得 ---
        self.num_of_hidden_nodes = pd.read_csv(io.StringIO(num_of_hidden_nodes), header=None, skipinitialspace=True).values[0].tolist()
        
        # --- 基底クラスの初期化 ---
        super().__init__(output_dir=output_dir, model_file=model_file,
                         web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo,
                         initializer=initializer, optimizer=optimizer, loss=loss,
                         dropout_rate=dropout_rate, learning_rate=learning_rate,
                         dataset_type=dataset_type, da_params=da_params,
                         batch_size=batch_size, epochs=epochs)
        
        # --- モデル構築 ---
        if (self.model is None):
            self.model = _load_model(input_shape)
            self._compile_model(optimizer=self.optimizer, loss=self.loss, init_lr=self.learning_rate)
            if (self.output_dir is not None):
                keras.utils.plot_model(self.model, Path(self.output_dir, 'plot_model.png'), show_shapes=True)
        
        return
    


#---------------------------------
# メイン処理; Trainerモジュールテスト
#---------------------------------
def main():
    import argparse
    def _argparse():
        parser = argparse.ArgumentParser(description='Trainerモジュールテスト\n'
                    '  * test_mode=\'ResNet\': ResNetのモデル構造確認(ResNet50の構造をTensorFlow公開モデルと比較)',
                    formatter_class=argparse.RawTextHelpFormatter)

        # --- 引数を追加 ---
        parser.add_argument('--test_mode', dest='test_mode', type=str, default='ResNet', required=False, \
                help='テストモード(ResNet)')

        args = parser.parse_args()
        return args

    # --- 引数処理 ---
    args = _argparse()
    print(args.test_mode)
    
    # --- モジュールテスト ---
    if (args.test_mode == 'ResNet'):
        trainer = TrainerResNet([224, 224, 3], 1000, output_dir=None, model_type='resnet50')
    else:
        print('[ERROR] Unknown test_mode: {}'.format(args.test_mode))
    
    return

    
if __name__ == '__main__':
    main()
'''