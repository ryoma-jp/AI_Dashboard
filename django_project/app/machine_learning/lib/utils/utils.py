"""Utilities

The common modules are described in this file.
"""

import os
import json
import shutil
import tarfile
import gzip
import cv2
import pandas as pd

from urllib import request
from pathlib import Path


def download_file(url, save_dir='output'):
    """Download file

    This function downloads file to the directory specified ``save_dir`` from ``url``.

    Args:
        url (string): specify URL
        save_dir (string): specify the directory to save file.
    """

    data = request.urlopen(url).read()
    with open(Path(save_dir, Path(url).name), 'wb') as f:
        f.write(data)

def safe_extract_tar(tar_file, path, members=None, *, numeric_owner=False):
    """Extract tarball (applied CVE-2007-4559 Patch)

    This function extracts tarball.

    Args:
        tar_file (string): tar.gz file
        path (string): the directory to files are extracted
    """
    def _is_within_directory(directory, target):
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    with tarfile.open(tar_file) as tar:
        for member in tar.getmembers():
            member_path = Path(path, member.name)
            if not _is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
            
        tar.extractall(path, members, numeric_owner=numeric_owner)

def safe_extract_gzip(gzip_file, path):
    """Extract gzip
    
    This function extracts gzip file.
    
    Args:
        gzip_file (string): gzip file
        path (string): the directory to files are extracted
    """

    with gzip.open(gzip_file, 'rb') as gzip_r:
        read_data = gzip_r.read()
        with open(Path(path, os.path.splitext(Path(gzip_file).name)[0]), 'wb') as gzip_w:
            gzip_w.write(read_data)

def zip_compress(zip_name, root_dir):
    """Compress files to zip
    
    This function compresses files specifed by ``files`` to the zip file.
    
    Args:
        zip_name (string): zip file name without extension
        root_dir (string): root directory to compress
    """
    
    shutil.make_archive(zip_name, 'zip', root_dir=root_dir)

def load_keys_from_meta(df_meta):
    """Load keys from meta
    
    This function loads keys from ``info.json`` of meta data.
    
    Args:
        df_meta (pandas.DataFrame): meta data
    
    Returns:
        list: keys
    """
    
    keys = []
    for key in df_meta['keys']:
        keys.append(key['name'])
    
    return keys

def save_meta(output_dir, is_analysis, task, input_type, keys):
    """Save Meta data (info.json)

    This function creates ``info.json``.

    Args:
        output_dir (string): specify the output directory
        is_analysis (string): meta parameter (see meta data apecification)
        task (string): meta parameter (see meta data apecification)
        input_type (string): meta parameter (see meta data apecification)
        keys (list): meta parameter (see meta data apecification)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    dict_meta = {
        'is_analysis': is_analysis,
        'task': task,
        'input_type': input_type,
        'keys': keys,
    }
    with open(Path(output_dir, 'info.json'), 'w') as f:
        json.dump(dict_meta, f, ensure_ascii=False, indent=4)

def save_image_files(images, labels, ids, output_dir, name='images', key_name='img_file', n_data=0):
    """Save Image Files

    This function creates and saves image files to ``output_dir``, and also creates ``info.json``.

    Args:
        images (numpy.ndarray): images (shape: [NHWC](or [NHW] if C=1), channel shape: [RGB])
        labels (numpy.ndarray): target labels (shape: [N])
        ids (list): sample identifiers (shape: [N])
        output_dir (string): specify the output directory
        name (string): specify the name of images and prefix
        key_name (string): specify the key name of info.json
        n_data (int): number of image files to save
    """

    os.makedirs(Path(output_dir, name), exist_ok=True)
    
    if ((n_data <= 0) or (n_data > len(images))):
        n_data = len(images)
        
    dict_image_file = []
    for (image, label, id) in zip(images[0:n_data], labels[0:n_data], ids[0:n_data]):
        image_file = str(Path(name, f'{id:08}.png'))
        if (len(image.shape) == 3):
            # --- RGB ---
            cv2.imwrite(str(Path(output_dir, image_file)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            # --- Gray scale ---
            cv2.imwrite(str(Path(output_dir, image_file)), image)
        
        dict_image_file.append({
            'id': str(id),
            key_name: image_file,
            'target': str(label),
        })
        
    # --- save image files information to json file ---
    with open(Path(output_dir, 'info.json'), 'w') as f:
        json.dump(dict_image_file, f, ensure_ascii=False, indent=4)
    
    return None

def save_table_info(df_meta, df_x, df_y, output_dir):
    """Save Table Information (info.json)
    
    This function creates ``info.json``.
    
    Args:
        df_meta (pandas.DataFrame): meta data
        df_x (pandas.DataFrame): input data
        df_y (pandas.DataFrame): target
        output_dir (string): directory path to ``info.json``
    
    """
    
    os.makedirs(output_dir, exist_ok=True)
    keys = load_keys_from_meta(df_meta)
    df_table_info = pd.concat([
                        pd.DataFrame({'id':[n for n in range(len(df_x))]}),
                        df_x.reset_index()[keys],
                        pd.DataFrame({'target':df_y.values.reshape(-1)})], axis=1)
    
    with open(Path(output_dir, 'info.json'), 'w') as f:
        json.dump(df_table_info.to_dict(orient='records'), f, ensure_ascii=False, indent=4)
    
    return None
    


