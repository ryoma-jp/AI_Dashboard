"""Main routine of Machine Learning for Training

Training model with AI Model SDK.
"""

#---------------------------------
# Import modules
#---------------------------------
import os
import sys
import argparse
import pickle
from pathlib import Path

from machine_learning.lib.data_loader.data_loader import DataLoaderPascalVOC2012
from machine_learning.lib.utils.utils import save_meta

#---------------------------------
# Functions
#---------------------------------
def ArgParser():
        """ArgParser

        Load arguments

        """

        parser = argparse.ArgumentParser(
                description='Training with AI Model SDK',
                formatter_class=argparse.RawTextHelpFormatter,
        )

        parser.add_argument(
                '--sdk_path', dest='sdk_path', type=str, default=None, required=True,
                help='AI Model SDK path',
        )
        parser.add_argument(
                '--dataset', dest='dataset', type=str, default=None, required=False,
                help='path to dataset.pkl\n'
                        '  - If set to None, load dataset to /tmp/dataset/dataset.pkl\n'
                        '  - select dataset type with --dataset_type',
        )
        parser.add_argument(
                '--dataset_type', dest='dataset_type', type=str, default='', required=False,
                choices=['CIFAR-10', 'MNIST', 'PascalVOC'],
                help='Type of dataset\n'
                        '  - CIFAR-10\n'
                        '  - MNIST\n'
                        '  - PascalVOC',
        )
        parser.add_argument(
                '--meta_json', dest='meta_json', type=str, default=None, required=False,
                help='info.json for meta data',
        )
        parser.add_argument(
                '--train_json', dest='train_json', type=str, default=None, required=False,
                help='info.json for train data',
        )
        parser.add_argument(
                '--val_json', dest='val_json', type=str, default=None, required=False,
                help='info.json for val data',
        )
        parser.add_argument(
                '--test_json', dest='test_json', type=str, default=None, required=False,
                help='info.json for test data',
        )
        parser.add_argument(
                '--model_path', dest='model_path', type=str, default=None, required=True,
                help='AI Model Path',
        )
        parser.add_argument(
                '--web_app_ctrl_fifo', dest='web_app_ctrl_fifo', type=str, default=None, required=False,
                help='FIFO path for Web App Control',
        )
        parser.add_argument(
                '--trainer_ctrl_fifo', dest='trainer_ctrl_fifo', type=str, default=None, required=False,
                help='FIFO path for Trainer Control',
        )

        args = parser.parse_args()

        return args

def main():
    """main
    
    Main function
    
    """
    args = ArgParser()
    print('[INFO] Arguments')
    print(f'  * args.sdk_path = {args.sdk_path}')
    print(f'  * args.dataset = {args.dataset}')
    print(f'  * args.dataset_type = {args.dataset_type}')
    print(f'  * args.meta_json = {args.meta_json}')
    print(f'  * args.train_json = {args.train_json}')
    print(f'  * args.val_json = {args.val_json}')
    print(f'  * args.test_json = {args.test_json}')
    print(f'  * args.model_path = {args.model_path}')
    print(f'  * args.web_app_ctrl_fifo = {args.web_app_ctrl_fifo}')
    print(f'  * args.trainer_ctrl_fifo = {args.trainer_ctrl_fifo}')
    
    # --- Check args.dataset ---
    if (args.dataset is None):
        print('[INFO] Create dataset.pkl')
        dataset_dir = '/tmp/dataset'
        os.makedirs(dataset_dir, exist_ok=True)

        if (args.dataset_type == 'PascalVOC'):
            # --- Create dataloader object ---
            dataloader = DataLoaderPascalVOC2012(dataset_dir, validation_split=0.2, download=True)

            # --- Create meta data ---
            meta_dir = Path(dataset_dir, 'meta')
            keys = [{
                    'name': 'img_file',
                    'type': 'image_file',
                    }]
            dict_meta = save_meta(meta_dir, 'True', 'object_detection', 'image_data', keys)
        
        # --- save dataset object to pickle file ---
        with open(Path(dataset_dir, 'dataset.pkl'), 'wb') as f:
            pickle.dump(dataloader, f)

        args.dataset = str(Path(dataset_dir, 'dataset.pkl'))
        print(f'[INFO] Create dataset.pkl done: {args.dataset}')

    # --- add AI Model SDK path to Python path ---
    sys.path.append(args.sdk_path)
    
    # --- import AI Model SDK ---
    from ai_model_sdk import AI_Model_SDK
    print(AI_Model_SDK.__version__)

    # --- Create instance ---
    dataset_params = {
        'meta': args.meta_json,
        'train': args.train_json,
        'val': args.val_json,
        'test': args.test_json,
    }
    model_params = {
        'model_path': args.model_path,
    }
    ai_model_sdk = AI_Model_SDK(args.dataset, model_params, web_app_ctrl_fifo=args.web_app_ctrl_fifo, trainer_ctrl_fifo=args.trainer_ctrl_fifo)

    # --- load dataset ---
    ai_model_sdk.load_dataset()

    # --- build model ---
    ai_model_sdk.build_model()

    # --- training ---
    ai_model_sdk.train_model()

    # --- save model ---
    ai_model_sdk.save_model()
    
    return

#---------------------------------
# main routine
#---------------------------------
if __name__ == '__main__':
    main()

