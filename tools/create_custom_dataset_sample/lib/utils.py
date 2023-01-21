"""Utilities

The common modules are described in this file.
"""

import os
import json
import zipfile
import cv2

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

def safe_extract(tar, path, members=None, *, numeric_owner=False):
    """Extract tarball (applied CVE-2007-4559 Patch)

    This function extracts tarball.

    Args:
        tar (string): tar.gz file
        path (string): the directory to files are extracted
    """
    def _is_within_directory(directory, target):
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    for member in tar.getmembers():
        member_path = Path(path, member.name)
        if not _is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
        
        tar.extractall(path, members, numeric_owner=numeric_owner)

def zip_compress(files, zip_name, path):
    """Compress files to zip
    
    This function compresses files specifed by ``files`` to the zip file.
    
    Args:
        files (list): file list to compress ([filepath, arcname])
        zip_name (string): zip file name
        path (string): save path
    """
    
    with zipfile.ZipFile(Path(path, zip_name), 'w',
                         compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zip:
        for (file, arcname) in files:
            zip.write(file, arcname=arcname)

def save_image_files(images, labels, output_dir, name='images', n_data=0):
    """Save Image Files

    This function creates and saves image files to ``output_dir``, and also creates "info.json".

    Args:
        images (numpy.ndarray): images (shape: [NHWC](or [NHW] if C=1), channel shape: [RGB])
        labels (numpy.ndarray): target labels (shape: [N])
        output_dir (string): specify the output directory
        name (string): specify the name of images and prefix
        n_data (int): number of image files to save
    """

    dict_image_file = {
        'id': [],
        'file': [],
        'class_id': [],
    }
    
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    
    if ((n_data <= 0) or (n_data > len(images))):
        n_data = len(images)
        
    for i, (image, label) in enumerate(zip(images[0:n_data], labels[0:n_data])):
        image_file = os.path.join(name, f'{i:08}.png')
        cv2.imwrite(os.path.join(output_dir, image_file), image)
        
        dict_image_file['id'].append(i)
        dict_image_file['file'].append(image_file)
        # dict_image_file['class_id'].append(int(np.argmax(label)))
        dict_image_file['class_id'].append(int(label))
        
    # --- save image files information to json file ---
    df_image_file = pd.DataFrame(dict_image_file)
    with open(os.path.join(output_dir, 'info.json'), 'w') as f:
        json.dump(json.loads(df_image_file.to_json(orient='records')), f, ensure_ascii=False, indent=4)
    
    return None




