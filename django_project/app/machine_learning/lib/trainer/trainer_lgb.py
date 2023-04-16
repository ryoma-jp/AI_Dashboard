#! -*- coding: utf-8 -*-

"""Trainer for LightGBM

This file describes the training and the prediction process of LightGBM.

"""

import os
import pickle
import json
import math
import lightgbm as lgb
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LogSummaryWriterCallback:
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
            

class TrainerLightGBM():
    """ Base Class of Trainer
    
    This class is the base class of LightGBM trainer.
    
    """
    
    def __init__(self, output_dir=None, model_file=None, 
            web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
            num_leaves=32, max_depth=4,
            learning_rate=0.001, feature_fraction=0.5,
            bagging_fraction=0.8, bagging_freq=0,
            lambda_l1=1.0, lambda_l2=5.0, boosting='gbdt'):
        """Constructor
        
        This function is constructor.
        
        """
        
        # --- Initialize ---
        self.input_tensor_name = ''
        self.output_tensor_name = ''
        
        # --- Load parameters ---
        self.web_app_ctrl_fifo = web_app_ctrl_fifo
        self.trainer_ctrl_fifo = trainer_ctrl_fifo
        
        # --- create output directory ---
        self.output_dir = output_dir
        if (self.output_dir is not None):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # --- load model ---
        if (model_file is not None):
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = None
        
        # --- set hyper parameters
        self.params = {
            'objective': 'regression',
            'metric': 'l1,l2,rmse',
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'min_data_in_leaf': 5,
            'learning_rate': learning_rate,
            'boosting': 'gbdt',
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'verbosity': -1,
            'random_state': 42,
            'early_stopping_rounds': 100,
        }
        print('[INFO]')
        for key in self.params.keys():
            print(f'  * {key}: {self.params[key]}')
        
        # --- summary writer ---
        self.writer = SummaryWriter(log_dir=Path(output_dir, 'logs'))

    def fit(self, x_train, y_train,
            x_val=None, y_val=None,
            x_test=None, y_test=None):
        """ Training
        
        This function runs the training model.
        
        Args:
            x_train (numpy.ndarray): training data (input)
            y_train (numpy.ndarray): training data (label)
            x_val (numpy.ndarray): validation data (input)
            y_val (numpy.ndarray): validation data (label)
            x_test (numpy.ndarray): test data (input)
            y_test (numpy.ndarray): test data (label)
        """
        
        # --- Create dataset ---
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_names = ['Train']
        valid_sets = [train_data]
        if ((x_val is not None) and (y_val is not None)):
            val_data = lgb.Dataset(x_val, label=y_val)
            valid_names.append('Validation')
            valid_sets.append(val_data)
        
        # --- Training ---
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
        
        # --- get and save metrics ---
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
        os.makedirs(Path(self.output_dir, 'metrics'), exist_ok=True)
        with open(Path(self.output_dir, 'metrics', 'metrics.json'), 'w') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        # --- send finish status to web app ---
        if (self.web_app_ctrl_fifo is not None):
            with open(self.web_app_ctrl_fifo, 'w') as f:
                f.write('trainer_done\n')
        
        # --- close writer ---
        self.writer.close()
        
    def save_model(self):
        """ Save Model
        
        This function saves the trained model.
        
        """
        # --- Create model directory ---
        model_dir = Path(self.output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # --- save model ---
        with open(Path(model_dir, 'lightgbm_model.pickle'), 'wb') as f:
            pickle.dump(self.model, f)
        
        return
    
    def predict(self, x):
        """ Predict
        
        This function predicts ``x``.
        
        """
        
        return self.model.predict(x)
    
    def get_importance(self, index=None):
        """ Get Importance
        
        This function returns feature importance as pandas.DataFrame
        
        Args:
            index (list): string list of features
        
        """
        
        return pd.DataFrame(self.model.feature_importance(), index=index, columns=['importance'])
        