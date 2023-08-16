
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from machine_learning.lib.utils.utils import save_config

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class AI_Model_SDK():
    """AI Model SDK
    
    Sample SDK for training classification CNN model using CIFAR-10 dataset
    """
    __version__ = 'SimpleCNN v0.0.1'

    class CustomCallback(keras.callbacks.Callback):
        def __init__(self, trainer_ctrl_fifo):
            super().__init__()
            self.trainer_ctrl_fifo = trainer_ctrl_fifo
            
        #def on_train_batch_end(self, batch, logs=None):
        #    if (self.trainer_ctrl_fifo is not None):
        #        fd = os.open(self.trainer_ctrl_fifo, os.O_RDONLY | os.O_NONBLOCK)
        #        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        #        flags &= ~os.O_NONBLOCK
        #        fcntl.fcntl(fd, fcntl.F_SETFL, flags)
        #        
        #        try:
        #            command = os.read(fd, 128)
        #            command = command.decode()[:-1]
        #            while (True):
        #                buf = os.read(fd, 65536)
        #                if not buf:
        #                    break
        #        finally:
        #            os.close(fd)
        #    
        #        if (command):
        #            if (command == 'stop'):
        #                print('End batch: recv command={}'.format(command))
        #                self.model.stop_training = True
        #            else:
        #                print('End batch: recv unknown command={}'.format(command))
            
        def on_epoch_end(self, epoch, logs=None):
            keys = list(logs.keys())
            log_str = ''
            for key in keys[:-1]:
                log_str += '({} = {}), '.format(key, logs[key])
            log_str += '({} = {})'.format(keys[-1], logs[keys[-1]])
            print("End epoch {}: {}".format(epoch, log_str))

    def __init__(self, dataset_params, model_params):
        """Constructor

        Args:
            dataset_params (dict) : dataset parameters
                                      - 'meta': info.json path for meta data
                                      - 'train': info.json path for train data
                                      - 'val': info.json path for validation data
                                      - 'test': info.json path for test data
            model_params (dict) : AI model parameters
                                    - 'model_path': path to save trained model
        """

        def split_input_and_target(dataset_params):
            """Split Data to Input and Target
            Split input samples and target from each ``info.json`` files
            """
            x_train = y_train = x_val = y_val = x_test = y_test = None

            if (('meta' in dataset_params.keys()) and (dataset_params['meta'] is not None)):
                df_meta = pd.read_json(dataset_params['meta'])
                input_key_list = [key['name'] for key in df_meta['keys']]

                if (('train' in dataset_params.keys()) and (dataset_params['train'] is not None)):
                    df_train = pd.read_json(dataset_params['train'])
                    x_train = df_train[['id'] + input_key_list]
                    y_train = df_train[['id', 'target']]

                print(dataset_params)
                if (('val' in dataset_params.keys()) and (dataset_params['val'] is not None)):
                    df_val = pd.read_json(dataset_params['val'])
                    x_val = df_val[['id'] + input_key_list]
                    y_val = df_val[['id', 'target']]

                if (('test' in dataset_params.keys()) and (dataset_params['test'] is not None)):
                    df_test = pd.read_json(dataset_params['test'])
                    x_test = df_test[['id'] + input_key_list]
                    y_test = df_test[['id', 'target']]

            return x_train, y_train, x_val, y_val, x_test, y_test
        
        # --- initialize parameters ---
        self.input_shape = [32, 32, 3]    # [H, W, C]
        self.class_num = 10
        self.model_path = model_params['model_path']
        self.trainer_ctrl_fifo = None

        # --- load info.json ---
        self.x_train_info, self.y_train_info, \
        self.x_val_info, self.y_val_info, \
        self.x_test_info, self.y_test_info \
            = split_input_and_target(dataset_params)

        # --- replace path to absolute path ---
        dataset_base_dir = Path(dataset_params['meta']).parent.parent
        if (self.x_train_info is not None):
            self.x_train_info['img_file'] = self.x_train_info['img_file'].map(lambda x: Path(dataset_base_dir, 'train', x))
        if (self.x_val_info is not None):
            self.x_val_info['img_file'] = self.x_val_info['img_file'].map(lambda x: Path(dataset_base_dir, 'validation', x))
        if (self.x_test_info is not None):
            self.x_test_info['img_file'] = self.x_test_info['img_file'].map(lambda x: Path(dataset_base_dir, 'test', x))

        # --- save config file ---
        configurable_parameters = []
        config_model = {
            'model': configurable_parameters,
        }
        save_config(config_model, self.model_path)

        return
    
    def preprocess_data(self, x):
        """Preprocess Data
        """

        # --- Normalization ---
        y = x / 255.0

        return y

    def load_dataset(self):
        """Load Dataset
        
        Load dataset from info and preprocess each samples
        """

        # --- input tensor ---
        fn_img2ndarray = lambda x: np.array(Image.open(x))
        if (self.x_train_info is None):
            self.x_train = None
        else:
            self.x_train = np.array([fn_img2ndarray(x) for x in self.x_train_info['img_file']])
            self.x_train = self.preprocess_data(self.x_train)

        if (self.x_val_info is None):
            self.x_val = None
        else:
            self.x_val = np.array([fn_img2ndarray(x) for x in self.x_val_info['img_file']])
            self.x_val = self.preprocess_data(self.x_val)

        if (self.x_test_info is None):
            self.x_test = None
        else:
            self.x_test = np.array([fn_img2ndarray(x) for x in self.x_test_info['img_file']])
            self.x_test = self.preprocess_data(self.x_test)

        # --- target ---
        if (self.y_train_info is None):
            self.y_train = None
        else:
            self.y_train = to_categorical(self.y_train_info['target'], self.class_num)

        if (self.y_val_info is None):
            self.y_val = None
        else:
            self.y_val = to_categorical(self.y_val_info['target'], self.class_num)

        if (self.y_test_info is None):
            self.y_test = None
        else:
            self.y_test = to_categorical(self.y_test_info['target'], self.class_num)

        return

    def build_model(self):
        """Build Model

        This is CNN sample model structured by Conv2D, ReLU and BatchNormalization.
        The original BatchNormalization paper prescribes using the BatchNormalization before ReLU.
        (https://arxiv.org/abs/1502.03167)

        But there is clarified that the after ReLU is better than the before ReLU by the after activities.
        (https://stackoverflow.com/questions/47143521/where-to-apply-batch-normalization-on-standard-cnns)
        """
        input = keras.layers.Input(shape=self.input_shape)
        
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=self.input_shape, padding='same')(input)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(512)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(512)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.BatchNormalization()(x)

        y = keras.layers.Dense(self.class_num, activation='softmax')(x)
        
        self.model = keras.models.Model(input, y)
        self.model.summary()

        return

    def train_model(self):
        """Train Model
        """
        # --- compile model ---
        learning_rate = 0.001
        weight_decay = 0.004
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        metrics = ['accuracy']
        loss = 'categorical_crossentropy'
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        # --- callbacks ---
        os.makedirs(Path(self.model_path, 'checkpoints'), exist_ok=True)
        checkpoint_path = Path(self.model_path, 'checkpoints', 'model.ckpt')
        cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
        custom_callback = self.CustomCallback(self.trainer_ctrl_fifo)
        tensorboard_logdir = Path(self.model_path, 'logs')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1)
        #callbacks = [cp_callback, es_callback]
        callbacks = [cp_callback, custom_callback, tensorboard_callback]

        # --- data augmentation ---
        datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            channel_shift_range=0.2,
            horizontal_flip=True)
        datagen.fit(self.x_train)
        
        # --- fit ---
        batch_size = 512
        epochs = 100
        history = self.model.fit(datagen.flow(self.x_train, self.y_train, batch_size=batch_size),
                    steps_per_epoch=len(self.x_train)//batch_size, validation_data=(self.x_val, self.y_val),
                    epochs=epochs, callbacks=callbacks,
                    verbose=0)
        
        return

    def Predict(self):
        """Predict
        """
        return
    
    def eval_model(self):
        """Evaluate Model
        """
        return
