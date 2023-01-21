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
from lib.utils import download_file, safe_extract, zip_compress, save_image_files

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
    
    # --- Initialize ---
    
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
    if (args.dataset_name == 'cifar-10'):
        meta_dir = Path(args.output_dir, 'meta')
        os.makedirs(meta_dir, exist_ok=True)
        
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
        with open(Path(meta_dir, 'info.json'), 'w') as f:
            json.dump(dict_meta, f, ensure_ascii=False, indent=4)
        
        zip_compress(Path(args.output_dir, 'meta'), meta_dir)
    
    # --- Create train.zip ---
    if (args.dataset_name == 'cifar-10'):
        train_dir = Path(args.output_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        
        # --- Create index list and shuffle for n_data ---
        idx_list = np.arange(len(train_images), dtype=np.int32)
        np.random.shuffle(idx_list)
        
        idx_list_shuffled = idx_list[0:min(args.n_data, len(train_images))]
        save_image_files(train_images[idx_list_shuffled],
                         train_labels[idx_list_shuffled],
                         idx_list_shuffled,
                         train_dir, name='images', key_name='img_file', n_data=args.n_data)
    
        zip_compress(Path(args.output_dir, 'train'), train_dir)
        
    # --- Create validation.zip ---
    if (args.dataset_name == 'cifar-10'):
        # --- T.B.D ---
        #  * CIFAR-10 dataset does not have validation data
        #  * but validation data is be able to create using ``sklearn.model_selection.train_test_split``
        pass
        
    # --- Create test.zip ---
    if (args.dataset_name == 'cifar-10'):
        test_dir = Path(args.output_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        
        # --- Create index list and shuffle for n_data ---
        idx_list = np.arange(len(test_images), dtype=np.int32)
        np.random.shuffle(idx_list)
        
        idx_list_shuffled = idx_list[0:min(args.n_data, len(test_images))]
        save_image_files(test_images[idx_list_shuffled],
                         test_labels[idx_list_shuffled],
                         idx_list_shuffled,
                         test_dir, name='images', key_name='img_file', n_data=args.n_data)
    
        zip_compress(Path(args.output_dir, 'test'), test_dir)
    
    return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
    main()

