
import os
import pickle
import numpy as np
import pandas as pd
import fcntl
from pathlib import Path
from PIL import Image
from machine_learning.lib.utils.utils import save_config
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from machine_learning.lib.data_loader.data_loader import load_dataset_from_tfrecord

class AI_Model_SDK():
    """AI Model SDK
    
    Sample SDK for training classification CNN model using CIFAR-10 dataset
    """
    __version__ = 'SimpleCNN for CIFAR-10 v0.0.1'

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

    def __init__(self, dataset_path, model_params, web_app_ctrl_fifo=None, trainer_ctrl_fifo=None):
        """Constructor

        Args:
            dataset_path (string) : file path of dataset.pkl (is DataLoader object)
                                    - attributes
                                        - train_dataset (dict)
                                        - validation_dataset (dict)
                                        - test_dataset (dict)
                                    - dict keys
                                        - 'tfrecord_path': path to tfrecord file
                                        - 'class_name_file_path': path to class name file
                                        - 'model_input_size': model input size
            model_params (dict) : AI model parameters
                                    - 'model_path': path to save trained model
        """

        def split_input_and_target(dataset_params):
            """Split Data to Input and Target
            Split input samples and target from each ``info.json`` files
            """
            x_train = y_train = x_val = y_val = x_test = y_test = x_inference = y_inference = None

            if (('meta' in dataset_params.keys()) and (dataset_params['meta'] is not None)):
                df_meta = pd.read_json(dataset_params['meta'])
                input_key_list = [key['name'] for key in df_meta['keys']]

                print(dataset_params)
                if (('train' in dataset_params.keys()) and (dataset_params['train'] is not None)):
                    df_train = pd.read_json(dataset_params['train'])
                    x_train = df_train[['id'] + input_key_list]
                    y_train = df_train[['id', 'target']]

                if (('val' in dataset_params.keys()) and (dataset_params['val'] is not None)):
                    df_val = pd.read_json(dataset_params['val'])
                    x_val = df_val[['id'] + input_key_list]
                    y_val = df_val[['id', 'target']]

                if (('test' in dataset_params.keys()) and (dataset_params['test'] is not None)):
                    df_test = pd.read_json(dataset_params['test'])
                    x_test = df_test[['id'] + input_key_list]
                    y_test = df_test[['id', 'target']]

                if (('inference' in dataset_params.keys()) and (dataset_params['inference'] is not None)):
                    df_inference = pd.read_json(dataset_params['inference'])
                    x_inference = df_inference[['id'] + input_key_list]
                    y_inference = df_inference[['id', 'target']]

            return x_train, y_train, x_val, y_val, x_test, y_test, x_inference, y_inference
        
        def transform_targets(y_train):
            # Ensure labels are 1-D int tensors for sparse crossentropy
            y_outs = tf.cast(tf.squeeze(y_train, axis=-1), tf.int32)
            return y_outs
        
        def transform_images(x_train, size):
            x_train = tf.image.resize(x_train, (size, size))
            x_train = x_train / 255
            return x_train
        
        # --- initialize parameters ---
        self.input_shape = [32, 32, 3]    # [H, W, C]
        self.batch_size = 512
        self.class_num = 10
        self.model_path = model_params['model_path']
        self.trainer_ctrl_fifo = None
        self.web_app_ctrl_fifo = web_app_ctrl_fifo
        self.trainer_ctrl_fifo = trainer_ctrl_fifo
        self.task = 'classification'
        self.decoded_preds = {}

        # --- Debug log helper ---
        os.makedirs(self.model_path, exist_ok=True)
        self._debug_log_path = Path(self.model_path, 'debug.log')
        Path(self._debug_log_path).touch(exist_ok=True)

        ## --- load info.json ---
        #self.x_train_info, self.y_train_info, \
        #self.x_val_info, self.y_val_info, \
        #self.x_test_info, self.y_test_info, \
        #self.x_inference_info, self.y_inference_info \
        #    = split_input_and_target(dataset_params)
#
        ## --- replace path to absolute path ---
        #if (self.x_train_info is not None):
        #    self.x_train_info['img_file'] = self.x_train_info['img_file'].map(lambda x: Path(Path(dataset_params['train']).parent, x))
        #if (self.x_val_info is not None):
        #    self.x_val_info['img_file'] = self.x_val_info['img_file'].map(lambda x: Path(Path(dataset_params['val']).parent, x))
        #if (self.x_test_info is not None):
        #    self.x_test_info['img_file'] = self.x_test_info['img_file'].map(lambda x: Path(Path(dataset_params['test']).parent, x))
        #if (self.x_inference_info is not None):
        #    self.x_inference_info['img_file'] = self.x_inference_info['img_file'].map(lambda x: Path(Path(dataset_params['inference']).parent, x))

        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        self.train_dataset = load_dataset_from_tfrecord(
            'classification',
            dataset.train_dataset['tfrecord_path'], 
            dataset.train_dataset['class_name_file_path'],
            dataset.train_dataset['model_input_size'])
        self.train_dataset = self.train_dataset.shuffle(buffer_size=512)
        self.train_dataset = self.train_dataset.batch(self.batch_size, drop_remainder=True)
        self.train_dataset = self.train_dataset.map(lambda x, y: (
            transform_images(x, dataset.train_dataset['model_input_size']),
            transform_targets(y)))
        self.train_dataset = self.train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        self.validation_dataset = load_dataset_from_tfrecord(
            'classification',
            dataset.validation_dataset['tfrecord_path'], 
            dataset.validation_dataset['class_name_file_path'],
            dataset.validation_dataset['model_input_size'])
#        self.validation_dataset = self.validation_dataset.shuffle(buffer_size=512)
        self.validation_dataset = self.validation_dataset.batch(self.batch_size, drop_remainder=True)
        self.validation_dataset = self.validation_dataset.map(lambda x, y: (
            transform_images(x, dataset.validation_dataset['model_input_size']),
            transform_targets(y)))
        self.validation_dataset = self.validation_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        
        self.test_dataset = load_dataset_from_tfrecord(
            'classification',
            dataset.test_dataset['tfrecord_path'], 
            dataset.test_dataset['class_name_file_path'],
            dataset.test_dataset['model_input_size'])
#        self.test_dataset = self.test_dataset.shuffle(buffer_size=512)
        self.test_dataset = self.test_dataset.batch(self.batch_size, drop_remainder=True)
        self.test_dataset = self.test_dataset.map(lambda x, y: (
            transform_images(x, dataset.test_dataset['model_input_size']),
            transform_targets(y)))
        self.test_dataset = self.test_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        
        # --- save config file ---
        configurable_parameters = []
        config_model = {
            'model': configurable_parameters,
        }
        save_config(config_model, self.model_path)

        return

    def _debug_log(self, msg):
        print(msg, flush=True)
        if self._debug_log_path is not None:
            with open(self._debug_log_path, 'a') as f:
                f.write(f"{msg}\n")

    def _debug_label_stats(self, y, name):
        y_np = y.numpy()
        self._debug_log(f'[DEBUG] {name}: shape={y_np.shape}, dtype={y_np.dtype}, min={y_np.min()}, max={y_np.max()}')
        flat = y_np.reshape(-1)
        unique, counts = np.unique(flat, return_counts=True)
        # shorten if too many classes
        max_items = 10
        if len(unique) > max_items:
            unique = unique[:max_items]
            counts = counts[:max_items]
            suffix = ' (truncated)'
        else:
            suffix = ''
        self._debug_log(f'[DEBUG] {name}: classes={unique.tolist()}, counts={counts.tolist()}{suffix}')

    def _debug_dataset_sample(self, dataset, name):
        for x_b, y_b in dataset.take(1):
            self._debug_log(f'[DEBUG] {name}: x_batch shape={x_b.shape}, y_batch shape={y_b.shape}')
            try:
                self._debug_label_stats(y_b, f'{name} labels')
            except Exception as e:
                self._debug_log(f'[DEBUG] {name}: label stats error: {e}')
            break

    def _manual_accuracy_dataset(self, dataset, name, max_batches=2):
        total = 0
        correct = 0
        batches = 0
        ds_iter = dataset if max_batches is None else dataset.take(max_batches)
        for x_b, y_b in ds_iter:
            preds = self.model.predict(x_b, verbose=0)
            pred_idx = np.argmax(preds, axis=1)
            y_np = y_b.numpy().reshape(-1)
            # trim in case shapes mismatch due to padding
            n = min(len(pred_idx), len(y_np))
            pred_idx = pred_idx[:n]
            y_np = y_np[:n]
            correct += int(np.sum(pred_idx == y_np))
            total += n
            batches += 1
        if total > 0:
            acc = correct / total
            self._debug_log(f'[DEBUG] Manual accuracy ({name}, {batches} batches, {total} samples): {acc:.4f}')
    
    def _debug_pred_vs_label(self, dataset, name, n_samples=8):
        for x_b, y_b in dataset.take(1):
            preds = self.model.predict(x_b, verbose=0)
            pred_idx = np.argmax(preds, axis=1)
            y_np = y_b.numpy().reshape(-1)
            n = min(n_samples, len(pred_idx), len(y_np))
            pairs = list(zip(pred_idx[:n].tolist(), y_np[:n].tolist()))
            self._debug_log(f'[DEBUG] {name} pred/label pairs (pred, label): {pairs}')
            break
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

        ## --- input tensor ---
        #fn_img2ndarray = lambda x: np.array(Image.open(x))
        #if (self.x_train_info is None):
        #    self.x_train = None
        #else:
        #    self.x_train = np.array([fn_img2ndarray(x) for x in self.x_train_info['img_file']])
        #    self.x_train = self.preprocess_data(self.x_train)
#
        #if (self.x_val_info is None):
        #    self.x_val = None
        #else:
        #    self.x_val = np.array([fn_img2ndarray(x) for x in self.x_val_info['img_file']])
        #    self.x_val = self.preprocess_data(self.x_val)
#
        #if (self.x_test_info is None):
        #    self.x_test = None
        #else:
        #    self.x_test = np.array([fn_img2ndarray(x) for x in self.x_test_info['img_file']])
        #    self.x_test = self.preprocess_data(self.x_test)
#
        #if (self.x_inference_info is None): 
        #    self.x_inference = None
        #else:
        #    self.x_inference = np.array([fn_img2ndarray(x) for x in self.x_inference_info['img_file']])
        #    self.x_inference = self.preprocess_data(self.x_inference)
#
        ## --- target ---
        #if (self.y_train_info is None):
        #    self.y_train = None
        #else:
        #    self.y_train = to_categorical(self.y_train_info['target'], self.class_num)
#
        #if (self.y_val_info is None):
        #    self.y_val = None
        #else:
        #    self.y_val = to_categorical(self.y_val_info['target'], self.class_num)
#
        #if (self.y_test_info is None):
        #    self.y_test = None
        #else:
        #    self.y_test = to_categorical(self.y_test_info['target'], self.class_num)
#
        #if (self.y_inference_info is None):
        #    self.y_inference = None
        #else:
        #    self.y_inference = to_categorical(self.y_inference_info['target'], self.class_num)

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

    def save_model(self):
        """Save Model

        Save trained model
        """

        # --- save model for saved_model ---
        save_path = Path(self.model_path, 'models', 'saved_model')
        os.makedirs(save_path, exist_ok=True)
        self.model.save(save_path)

        # --- save model for h5 ---
        save_path = Path(self.model_path, 'models', 'h5', 'model.h5')
        os.makedirs(Path(self.model_path, 'h5'), exist_ok=True)
        self.model.save(save_path)

        return
    
    def load_model(self, trained_model_path):
        """Load Model

        Load trained model

        Args:
            trained_model_path (str) : path to trained model
        """

        self.model = keras.models.load_model(Path(trained_model_path, 'h5', 'model.h5'))
        self.model.summary()

        return
    
    def train_model(self):
        """Train Model
        """
        # --- compile model ---
        learning_rate = 0.001
        weight_decay = 0.0005
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        metrics = ['accuracy']
        loss = 'sparse_categorical_crossentropy'
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

        # --- Debug dataset samples before training ---
        self._debug_dataset_sample(self.train_dataset, 'train')
        self._debug_dataset_sample(self.validation_dataset, 'val')

        # --- data augmentation ---
        #datagen = ImageDataGenerator(
        #    rotation_range=0.2,
        #    width_shift_range=0.2,
        #    height_shift_range=0.2,
        #    zoom_range=0.2,
        #    channel_shift_range=0.2,
        #    horizontal_flip=True)
        #datagen.fit(self.x_train)
        
        # --- fit ---
        epochs = 100
        #history = self.model.fit(datagen.flow(self.x_train, self.y_train, batch_size=batch_size),
        #            steps_per_epoch=len(self.x_train)//batch_size, validation_data=(self.x_val, self.y_val),
        #            epochs=epochs, callbacks=callbacks,
        #            verbose=0)
        print(self.validation_dataset)
        history = self.model.fit(self.train_dataset,
                    validation_data=self.validation_dataset,
                    epochs=epochs, callbacks=callbacks,
                    verbose=0)

        # --- Manual accuracy check on small subsets after training ---
        self._manual_accuracy_dataset(self.train_dataset, 'train')
        self._manual_accuracy_dataset(self.validation_dataset, 'val')

        # --- Manual accuracy on full datasets ---
        self._manual_accuracy_dataset(self.train_dataset, 'train_full', max_batches=None)
        self._manual_accuracy_dataset(self.validation_dataset, 'val_full', max_batches=None)

        # --- Sample prediction vs label pairs ---
        self._debug_pred_vs_label(self.train_dataset, 'train')
        self._debug_pred_vs_label(self.validation_dataset, 'val')

        # --- Compare Keras evaluate vs manual accuracy on full datasets ---
        try:
            train_eval = self.model.evaluate(self.train_dataset, verbose=0)
            val_eval = self.model.evaluate(self.validation_dataset, verbose=0)
            # evaluate returns [loss, acc] when metrics=['accuracy']
            if isinstance(train_eval, (list, tuple)) and len(train_eval) >= 2:
                self._debug_log(f'[DEBUG] Keras evaluate train: loss={train_eval[0]:.4f}, acc={train_eval[1]:.4f}')
            else:
                self._debug_log(f'[DEBUG] Keras evaluate train: {train_eval}')
            if isinstance(val_eval, (list, tuple)) and len(val_eval) >= 2:
                self._debug_log(f'[DEBUG] Keras evaluate val: loss={val_eval[0]:.4f}, acc={val_eval[1]:.4f}')
            else:
                self._debug_log(f'[DEBUG] Keras evaluate val: {val_eval}')
        except Exception as e:
            self._debug_log(f'[DEBUG] Keras evaluate error: {e}')
        
        # --- Notice the finish training to Web app ---
        if (self.web_app_ctrl_fifo is not None):
            with open(self.web_app_ctrl_fifo, 'w') as f:
                f.write('trainer_done\n')
                
        return

    def predict(self, x, preprocessing=True):
        """Predict

        Predict target from input

        Args:
            x (numpy.ndarray) : input data
            preprocessing (bool) : if nesessary the preprocessing, set to True
        """

        if (preprocessing):
            x = self.preprocess_data(x)
        y = self.model.predict(x)
        
        return y
    
    def decode_prediction(self, pred):
        self.decoded_preds['detection_classes'] = np.argmax(pred, axis=1)
        return pred
    
    def eval_model(self, pred, target):
        """Evaluate Model

        Calculate accuracy score between pred and target

        Args:
            pred (numpy.ndarray): prediction
            target (numpy.ndarray): target
        """

        accuracy = accuracy_score(np.argmax(target, axis=1), np.argmax(pred, axis=1))
        ret = {'accuracy': accuracy}

        return ret
