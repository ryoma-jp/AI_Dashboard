#! /bin/bash

DATASET_DIR="${PWD}/test/simple_cnn/dataset"

# --- Prepare dataset ---
DATASET_DIR=${DATASET_DIR} python << 'EOT'
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from machine_learning.lib.data_loader.data_loader import DataLoaderCIFAR10
from machine_learning.lib.utils.utils import save_meta, save_image_files, save_table_info

dataset_dir = os.environ['DATASET_DIR']

cifar10_path = Path(dataset_dir, 'cifar-10-batches-py')
download = False
if (not cifar10_path.exists()):
    download = True

# --- load dataset ---
dataloader = DataLoaderCIFAR10(dataset_dir, validation_split=0.2, one_hot=False, download=download)

# --- save meta data ---
meta_dir = Path(dataset_dir, 'meta')
keys = [{
           'name': 'img_file',
           'type': 'image_file',
       }]
save_meta(meta_dir, 'True', 'classification', 'image_data', keys)

# --- load key_name from meta data as pandas.DataFrame ---
df_meta = pd.read_json(Path(dataset_dir, 'meta', 'info.json'), typ='series')

# --- get key_name ---
for key in df_meta['keys']:
    if (key['type'] == 'image_file'):
        key_name = key['name']
        break

# --- save image files ---
if (dataloader.train_x is not None):
    ids = np.arange(len(dataloader.train_x))
    save_image_files(dataloader.train_x, dataloader.train_y, ids,
                     Path(dataset_dir, 'train'), name='images', key_name=key_name)
if (dataloader.validation_x is not None):
    ids = np.arange(len(dataloader.validation_x))
    save_image_files(dataloader.validation_x, dataloader.validation_y, ids,
                     Path(dataset_dir, 'validation'), name='images', key_name=key_name)
if (dataloader.test_x is not None):
    ids = np.arange(len(dataloader.test_x))
    save_image_files(dataloader.test_x, dataloader.test_y, ids,
                     Path(dataset_dir, 'test'), name='images', key_name=key_name)


# --- save dataset object to pickle file ---
with open(Path(dataset_dir, 'dataset.pkl'), 'wb') as f:
    pickle.dump(dataloader, f)

EOT

# --- Training ---
python main.py --mode "train" --config "${PWD}/test/simple_cnn/config.json"

