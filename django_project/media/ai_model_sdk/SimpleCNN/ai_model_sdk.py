
import pandas as pd
from pathlib import Path
from machine_learning.lib.utils.utils import save_config

class AI_Model_SDK():
    """AI Model SDK
    """
    __version__ = 'SimpleCNN v0.0.1'

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

        self.model_path = model_params['model_path']
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = split_input_and_target(dataset_params)

        # --- save config file ---
        configurable_parameters = []
        config_model = {
            'model': configurable_parameters,
        }
        save_config(config_model, self.model_path)

        return
    
    def preprocess_data(self):
        """Preprocess Data
        """
        return

    def load_dataset(self):
        """Load Dataset
        """
        return

    def build_model(self):
        """Build Model
        """
        return

    def train_model(self):
        """Train Model
        """
        return

    def preprocess_data(self):
        """Preprocess Data
        """
        return

    def Predict(self):
        """Predict
        """
        return
    
    def eval_model(self):
        """Evaluate Model
        """
        return
