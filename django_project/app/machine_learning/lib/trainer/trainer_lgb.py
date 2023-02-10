#! -*- coding: utf-8 -*-

"""Trainer for LightGBM

This file describes the training and the prediction process of LightGBM.

"""

import os
import pickle
import lightgbm as lgb
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

class LogSummaryWriterCallback:
    """ Writing log callable class
    """
    
    def __init__(self, period=1, writer=None):
        self.period = period
        self.writer = writer
    
    def __call__(self, env):
        if (self.period > 0) and (env.evaluation_result_list) and (((env.iteration+1) % self.period)==0):
            if (self.writer is not None):
                for (name, metric, value, is_higher_better) in env.evaluation_result_list:
                    self.writer.add_scalar(f'{name}'s {metric}', value, env.iteration+1)
            else:
                print(env.evaluation_result_list)
            

class TrainerLightGBM():
    """ Base Class of Trainer
    
    This class is the base class of LightGBM trainer.
    
    """
    
    def __init__(self, output_dir=None, model_file=None, learning_rate=0.001):
        """Constructor
        
        This function is constructor.
        
        """
        
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
            'metric': 'mae',
            'num_leaves': 32,
            'max_depth': 4,
            'feature_fraction': 0.5,
            'subsample_freq': 1,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 5,
            'learning_rate': learning_rate,
            'boosting': 'gbdt',
            'lambda_l1': 1,
            'lambda_l2': 5,
            'verbosity': -1,
            'random_state': 42,
            'early_stopping_rounds': 100,
        }
        
        # --- summary writer ---
        self.writer = SummaryWriter(log_dir=Path(output_dir, 'logs'))

    def fit(self, x_train, y_train,
            x_val=None, y_val=None, x_test=None, y_test=None,
            web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
            da_params=None,
            batch_size=32, epochs=200,
            verbose=0):
        """ Training
        
        This function runs the training model.
        
        Args:
            x_train (numpy.ndarray): training data (input)
            y_train (numpy.ndarray): training data (label)
            x_val (numpy.ndarray): validation data (input)
            y_val (numpy.ndarray): validation data (label)
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
        
        # --- send finish status to web app ---
        if (web_app_ctrl_fifo is not None):
            with open(web_app_ctrl_fifo, 'w') as f:
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
    
    
