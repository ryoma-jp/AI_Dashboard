"""Create Custom Dataset

sample code for create dataset

"""

import os
import tarfile
import argparse
import json
import pickle
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from lib.data_sample_loader import load_cifar10_dataset
from lib.utils import download_file, safe_extract, zip_compress

def ArgParser():
    """Argument Parser
    """
    parser = argparse.ArgumentParser(description='This tool convert from open dataset to AI Dashboard\'s custom dataset format',
                formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--dataset_name', dest='dataset_name', type=str, default='cifar-10', required=False, \
            help='specifies the open dataset name (cifar-10 or mnist)')
    parser.add_argument('--input_dir', dest='input_dir', type=str, default='input', required=False, \
            help='input directory if necessary the download')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='output', required=False, \
            help='output directory')
    parser.add_argument('--n_data', dest='n_data', type=int, default=0, required=False, \
            help='number of data samples (if set to less than 0, get all samples)')

    args = parser.parse_args()

    return args

def main():
    """Main Function
    """
    
    # --- Load Arguments ---
    args = ArgParser()
    print('args.dataset_name : {}'.format(args.dataset_name))
    print('args.input_dir : {}'.format(args.input_dir))
    print('args.output_dir : {}'.format(args.output_dir))
    print('args.n_data : {}'.format(args.n_data))
    
    # --- Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Get Data Samples ---
    if (args.dataset_name == 'cifar-10'):
        tar_file = Path(args.output_dir, 'cifar-10-python.tar.gz')
        extracted_dir = Path(args.output_dir, 'cifar-10-batches-py')
        
        # --- Download if file is not exist ---
        if (not tar_file.exists()):
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            download_file(url, args.output_dir)
        
            # --- Extract taball ---
            with tarfile.open(tar_file) as tar:
                safe_extract(tar, path=args.output_dir)
        
        # --- Load samples ---
        train_images, train_labels, test_images, test_labels = load_cifar10_dataset(extracted_dir)
        print(f'train_images.shape = {train_images.shape}')
        print(f'train_labels.shape = {train_labels.shape}')
        print(f'test_images.shape = {test_images.shape}')
        print(f'test_labels.shape = {test_labels.shape}')
    
    # --- Create meta.zip ---
    dict_meta = {
        'is_analysis': 'True',
        'task': 'classification',
        'input_type': 'image_data',
        'keys': [
            {
                'name': 'img_file',
                'type': 'image_file',
            },
        ],
    }
    with open(Path(args.output_dir, 'info.json'), 'w') as f:
        json.dump(dict_meta, f, ensure_ascii=False, indent=4)
    
    meta_files = [
        [str(Path(args.output_dir, 'info.json')), 'info.json'],
    ]
    zip_compress(meta_files, 'meta.zip', args.output_dir)
    
    # --- Create train.zip ---
    
    # --- Create validation.zip ---
    
    # --- Create test.zip ---
    
    return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
    main()

