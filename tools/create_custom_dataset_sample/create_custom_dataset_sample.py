"""Create Custom Dataset

sample code for create dataset

"""

import os
import argparse
import json
import pickle
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from lib.data_sample_loader import load_mnist_dataset, load_cifar10_dataset
from machine_learning.lib.utils.utils import download_file, safe_extract_tar, safe_extract_gzip, zip_compress, save_image_files


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
    if (args.dataset_name == 'mnist'):
        mnist_url_list = [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        ]
        
        # --- Download if file is not exist ---
        for mnist_url in mnist_url_list:
            gzip_file = Path(args.output_dir, Path(mnist_url).name)
            if (not gzip_file.exists()):
                download_file(mnist_url, args.output_dir)
            
                # --- Extract gzip ---
                safe_extract_gzip(gzip_file, path=args.output_dir)
        
        # --- Load samples ---
        extracted_dir = args.output_dir
        train_images, train_labels, test_images, test_labels = load_mnist_dataset(extracted_dir)
        print(f'train_images.shape = {train_images.shape}')
        print(f'train_labels.shape = {train_labels.shape}')
        print(f'test_images.shape = {test_images.shape}')
        print(f'test_labels.shape = {test_labels.shape}')
        
    elif (args.dataset_name == 'cifar-10'):
        # --- Download if file is not exist ---
        tar_file = Path(args.output_dir, 'cifar-10-python.tar.gz')
        if (not tar_file.exists()):
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            download_file(url, args.output_dir)
        
            # --- Extract taball ---
            safe_extract_tar(tar_file, path=args.output_dir)
        
        # --- Load samples ---
        extracted_dir = Path(args.output_dir, 'cifar-10-batches-py')
        train_images, train_labels, test_images, test_labels = load_cifar10_dataset(extracted_dir)
        print(f'train_images.shape = {train_images.shape}')
        print(f'train_labels.shape = {train_labels.shape}')
        print(f'test_images.shape = {test_images.shape}')
        print(f'test_labels.shape = {test_labels.shape}')
    else:
        raise Exception(f'Unknown dataset name: {args.dataset_name}')
        
    
    # --- Create meta.zip ---
    if ((args.dataset_name == 'mnist') or (args.dataset_name == 'cifar-10')):
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
    else:
        raise Exception(f'Unknown dataset name: {args.dataset_name}')
    
    # --- Create train.zip ---
    if ((args.dataset_name == 'mnist') or (args.dataset_name == 'cifar-10')):
        train_dir = Path(args.output_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        
        # --- Create index list and shuffle for n_data ---
        idx_list = np.arange(len(train_images))
        np.random.shuffle(idx_list)
        
        n_data = args.n_data
        if ((n_data <= 0) or (n_data > len(train_images))):
            n_data = len(train_images)
        idx_list_shuffled = idx_list[0:min(n_data, len(train_images))]
        save_image_files(train_images[idx_list_shuffled],
                         train_labels[idx_list_shuffled],
                         idx_list_shuffled,
                         train_dir, name='images', key_name='img_file', n_data=n_data)
    
        zip_compress(Path(args.output_dir, 'train'), train_dir)
    else:
        raise Exception(f'Unknown dataset name: {args.dataset_name}')
        
    # --- Create validation.zip ---
    if ((args.dataset_name == 'mnist') or (args.dataset_name == 'cifar-10')):
        # --- T.B.D ---
        #  * MNIST and CIFAR-10 dataset don't have validation data
        #  * but validation data is be able to create using ``sklearn.model_selection.train_test_split``
        pass
    else:
        raise Exception(f'Unknown dataset name: {args.dataset_name}')
        
    # --- Create test.zip ---
    if ((args.dataset_name == 'mnist') or (args.dataset_name == 'cifar-10')):
        test_dir = Path(args.output_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        
        # --- Create index list and shuffle for n_data ---
        idx_list = np.arange(len(test_images), dtype=np.int32)
        np.random.shuffle(idx_list)
        
        n_data = args.n_data
        if ((n_data <= 0) or (n_data > len(test_images))):
            n_data = len(test_images)
        idx_list_shuffled = idx_list[0:min(n_data, len(test_images))]
        save_image_files(test_images[idx_list_shuffled],
                         test_labels[idx_list_shuffled],
                         idx_list_shuffled,
                         test_dir, name='images', key_name='img_file', n_data=n_data)
    
        zip_compress(Path(args.output_dir, 'test'), test_dir)
    else:
        raise Exception(f'Unknown dataset name: {args.dataset_name}')
    
    return

if __name__ == '__main__':
    main()

