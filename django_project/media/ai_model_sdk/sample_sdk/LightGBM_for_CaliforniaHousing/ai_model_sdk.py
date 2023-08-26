
import os
import pickle
from statistics import mean
import numpy as np
import pandas as pd
import fcntl
import lightgbm as lgb
from pathlib import Path
from PIL import Image
from machine_learning.lib.utils.utils import save_config
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error

class LogSummaryWriterCallback():
    """ Writing log callable class
    """
    
    def __init__(self, period=1, writer=None):
        self.period = period
        self.writer = writer
    
    def __call__(self, env):
        if (self.period > 0) and (env.evaluation_result_list) and (((env.iteration+1) % self.period)==0):
            if (self.writer is not None):
                scalars = {}
                for (name, metric, value, is_higher_better) in env.evaluation_result_list:
                    if (metric not in scalars.keys()):
                        scalars[metric] = {}
                    scalars[metric][name] = value
                    
                for key in scalars.keys():
                    self.writer.add_scalars(key, scalars[key], env.iteration+1)
            else:
                print(env.evaluation_result_list)

class AI_Model_SDK():
    """AI Model SDK
    
    Sample SDK for training LightGBM model using CaliforniaHousing dataset
    """
    __version__ = 'LightGBM for CaliforniaHousing v0.0.1'
            
    def __init__(self, dataset_params, model_params, web_app_ctrl_fifo=None, trainer_ctrl_fifo=None):
        """Constructor

        Args:
            dataset_params (dict) : dataset parameters
                                      - 'meta': info.json path for meta data
                                      - 'train': info.json path for train data
                                      - 'val': info.json path for validation data
                                      - 'test': info.json path for test data
                                      - 'inference': info.json path for inference data that doesn't need to have target
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
        
        # --- initialize parameters ---
        self.model_path = model_params['model_path']
        self.web_app_ctrl_fifo = web_app_ctrl_fifo
        self.trainer_ctrl_fifo = trainer_ctrl_fifo

        # --- load info.json ---
        self.x_train_info, self.y_train_info, \
        self.x_val_info, self.y_val_info, \
        self.x_test_info, self.y_test_info, \
        self.x_inference_info, self.y_inference_info \
            = split_input_and_target(dataset_params)

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

        # --- T.B.D ---
        y = x

        return y

    def load_dataset(self):
        """Load Dataset
        
        Load dataset from info and preprocess each samples
        """

        # --- input tensor ---
        if (self.x_train_info is None):
            self.x_train = None
        else:
            self.x_train = self.x_train_info.drop(['id'], axis=1).values
            self.x_train = self.preprocess_data(self.x_train)

        if (self.x_val_info is None):
            self.x_val = None
        else:
            self.x_val = self.x_val_info.drop(['id'], axis=1).values
            self.x_val = self.preprocess_data(self.x_val)

        if (self.x_test_info is None):
            self.x_test = None
        else:
            self.x_test = self.x_test_info.drop(['id'], axis=1).values
            self.x_test = self.preprocess_data(self.x_test)

        if (self.x_inference_info is None): 
            self.x_inference = None
        else:
            self.x_inference = self.x_inference_info.drop(['id'], axis=1).values
            self.x_inference = self.preprocess_data(self.x_inference)

        # --- target ---
        if (self.y_train_info is None):
            self.y_train = None
        else:
            self.y_train = self.y_train_info['target'].values

        if (self.y_val_info is None):
            self.y_val = None
        else:
            self.y_val = self.y_val_info['target'].values

        if (self.y_test_info is None):
            self.y_test = None
        else:
            self.y_test = self.y_test_info['target'].values

        if (self.y_inference_info is None):
            self.y_inference = None
        else:
            self.y_inference = self.y_inference_info['target'].values

        return

    def build_model(self):
        """Build Model

        No processing
        """

        return

    def save_model(self):
        """Save Model

        Save trained model
        """

        # --- save model to pkl ---
        save_path = Path(self.model_path, 'models', 'lightgbm.pkl')
        os.makedirs(Path(self.model_path, 'models'), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)

        return
    
    def load_model(self, trained_model_path):
        """Load Model

        Load trained model

        Args:
            trained_model_path (str) : path to trained model
        """

        with open(Path(trained_model_path, 'lightgbm.pkl'), 'rb') as f:
            self.model = pickle.load(f)

        return
    
    def train_model(self):
        """Train Model
        """
        # --- set hyper parameters ---
        self.params = {
            'objective': 'regression',
            'metric': 'l1,l2,rmse',
            'num_leaves': 32,
            'max_depth': 8,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'min_data_in_leaf': 5,
            'learning_rate': 0.01,
            'boosting': 'gbdt',
            'lambda_l1': 0.0,
            'lambda_l2': 0.0,
            'verbosity': -1,
            'random_state': 42,
            'early_stopping_rounds': 100,
        }
        print('[INFO]')
        for key in self.params.keys():
            print(f'  * {key}: {self.params[key]}')

        # --- summary writer ---
        self.writer = SummaryWriter(log_dir=Path(self.model_path, 'logs'))

        # --- create dataset ---
        train_data = lgb.Dataset(self.x_train, label=self.y_train)
        valid_names = ['Train']
        valid_sets = [train_data]
        if ((self.x_val is not None) and (self.y_val is not None)):
            val_data = lgb.Dataset(self.x_val, label=self.y_val)
            valid_names.append('Validation')
            valid_sets.append(val_data)
        
        # --- training ---
        self.model = lgb.train(
            self.params,
            train_data,
            valid_names=valid_names,
            valid_sets=valid_sets,
            num_boost_round=50000,
            callbacks=[
                lgb.log_evaluation(period=100),
                LogSummaryWriterCallback(period=100, writer=self.writer)
            ]
        )
        
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
    
    def eval_model(self, pred, target):
        """Evaluate Model

        Calculate mean squared error between pred and target

        Args:
            pred (numpy.ndarray): prediction
            target (numpy.ndarray): target
        """

        mse = mean_squared_error(target, pred)
        ret = {'MSE': mse}

        return ret
