import os
import fcntl
import logging
import json
import pickle
import shutil
import numpy as np
import pandas as pd

from pathlib import Path

from django.conf import settings

from app.models import Project, MlModel

from machine_learning.lib.data_loader.data_loader import DataLoaderCIFAR10
from machine_learning.lib.data_loader.data_loader import DataLoaderMNIST
from machine_learning.lib.data_loader.data_loader import DataLoaderCaliforniaHousing
from machine_learning.lib.data_loader.data_loader import DataLoaderCOCO2017
from machine_learning.lib.data_loader.data_loader import DataLoaderPascalVOC2012
from machine_learning.lib.data_loader.data_loader import DataLoaderCustom
from machine_learning.lib.utils.utils import save_meta, save_image_info, save_image_files, save_table_info

# Create your views here.

class SidebarActiveStatus():
    """ Class: Sidebar active status
    """
    def __init__(self):
        self.index = ''
        self.dataset = ''
        self.training = ''
        self.inference = ''
        self.view_streaming = ''


def get_version():
    """ Function: get_version
     * return version text
    """
    if (settings.DEBUG):
        return 'Debug mode'
    else:
        return '[T.B.D] VerX.XX'

def get_recv_fifo_command(fifo):
    """ Function: get_recv_fifo_command
     * return recieved command
    """
    fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    flags &= ~os.O_NONBLOCK
    fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    try:
        command = os.read(fd, 128)
        command = command.decode()[:-1]
        while (True):
            buf = os.read(fd, 65536)
            if not buf:
                break
    finally:
        os.close(fd)

    if (command):
        if (command == 'trainer_done'):
            logging.debug(f'{command} command recieved')
            return command
        else:
            logging.debug(f'Unknown command({command}) recieved')
            return None
    
    return None

def get_all_fifo_command():
    """ Function: get_all_fifo_command
     * get all fifo command
    """
    projects = Project.objects.all().order_by('-id').reverse()
    for project in projects:
        models = MlModel.objects.filter(project=project).order_by('-id').reverse()
        
        for model in models:
            with open(Path(model.model_dir, 'config.json'), 'r') as f:
                dict_config = json.load(f)
            
            while (True):
                recv_command = get_recv_fifo_command(dict_config['env']['web_app_ctrl_fifo']['value'])
                if (recv_command == 'trainer_done'):
                    model.training_pid = None
                    model.status = model.STAT_DONE
                    model.save()
                else:
                    break

def load_dataset(dataset):
    """Load Dataset
    
    Load dataset and return the class object
    
    Args:
        dataset: Dataset class object
    
    Return:
        Dataset class object
    """
    
    # --- load dataset ---
    dataset_dir = Path(settings.MEDIA_ROOT, settings.DATASET_DIR, dataset.project.hash)
    download_dir = Path(dataset_dir, f'dataset_{dataset.id}')
    if (download_dir.exists()):
        download = False
    else:
        download = True
        dataset.download_status = dataset.STATUS_PROCESSING
        dataset.save()
    
    if (dataset.name == 'MNIST'):
        # --- Create dataloader object ---
        dataloader = DataLoaderMNIST(download_dir, validation_split=0.2, one_hot=False, download=download)
        
        # --- Create meta data ---
        meta_dir = Path(download_dir, 'meta')
        keys = [{
                    'name': 'img_file',
                    'type': 'image_file',
                }]
        dict_meta = save_meta(meta_dir, 'True', 'classification', 'image_data', keys)
        
        # --- save image files ---
        if (dataloader.train_x is not None):
            ids = np.arange(len(dataloader.train_x))
            save_image_files(dataloader.train_x, dataloader.train_y, ids,
                             Path(download_dir, 'train'), name='images', key_name=keys[0]['name'])
        if (dataloader.validation_x is not None):
            ids = np.arange(len(dataloader.validation_x))
            save_image_files(dataloader.validation_x, dataloader.validation_y, ids,
                             Path(download_dir, 'validation'), name='images', key_name=keys[0]['name'])
        if (dataloader.test_x is not None):
            ids = np.arange(len(dataloader.test_x))
            save_image_files(dataloader.test_x, dataloader.test_y, ids,
                             Path(download_dir, 'test'), name='images', key_name=keys[0]['name'])
        
    elif (dataset.name == 'CIFAR-10'):
        # --- Create dataloader object ---
        dataloader = DataLoaderCIFAR10(download_dir, validation_split=0.2, one_hot=False, download=download)
        
        # --- Create meta data ---
        meta_dir = Path(download_dir, 'meta')
        keys = [{
                    'name': 'img_file',
                    'type': 'image_file',
                }]
        dict_meta = save_meta(meta_dir, 'True', 'classification', 'image_data', keys)
        
        # --- save image files ---
        if (dataloader.train_x is not None):
            ids = np.arange(len(dataloader.train_x))
            save_image_files(dataloader.train_x, dataloader.train_y, ids,
                             Path(download_dir, 'train'), name='images', key_name=keys[0]['name'])
        if (dataloader.validation_x is not None):
            ids = np.arange(len(dataloader.validation_x))
            save_image_files(dataloader.validation_x, dataloader.validation_y, ids,
                             Path(download_dir, 'validation'), name='images', key_name=keys[0]['name'])
        if (dataloader.test_x is not None):
            ids = np.arange(len(dataloader.test_x))
            save_image_files(dataloader.test_x, dataloader.test_y, ids,
                             Path(download_dir, 'test'), name='images', key_name=keys[0]['name'])
        
    elif (dataset.name == 'COCO2017'):
        # --- Create dataloader object ---
        dataloader = DataLoaderCOCO2017(download_dir, validation_split=0.2, download=download)
        
        # --- Create meta data ---
        meta_dir = Path(download_dir, 'meta')
        keys = [{
                    'name': 'img_file',
                    'type': 'image_file',
                }]
        dict_meta = save_meta(meta_dir, 'True', 'object_detection', 'image_data', keys)
        
        # --- save info.json (train) ---
        os.makedirs(Path(download_dir, 'train', 'images'), exist_ok=True)
        dict_image_file = []
        image_ids = dataloader.df_instances_train['image_id'].unique()
        file_names = dataloader.df_instances_train['file_name'].unique()
        instance_ids = dataloader.df_instances_train.groupby(by=['image_id'], sort=False)['instance_id'].apply(list).values
        bboxes = dataloader.df_instances_train.groupby(by=['image_id'], sort=False)['bbox'].apply(list).values
        category_ids = dataloader.df_instances_train.groupby(by=['image_id'], sort=False)['category_id'].apply(list).values
        category_names = dataloader.df_instances_train.groupby(by=['image_id'], sort=False)['category_name'].apply(list).values
        for id, image_file, instance_id, bbox, category_id, category_name in zip(image_ids, file_names, instance_ids, bboxes, category_ids, category_names):
            # --- copy from COCO directory(src) to AI Dashboard directory(dst)
            if (Path(download_dir, 'train2017', image_file).exists()):
                src_file = str(Path(download_dir, 'train2017', image_file))
                dst_file = str(Path(download_dir, 'train', 'images', image_file))
                shutil.move(src_file, dst_file)
            
            # --- set image info ---
            dict_image_file.append({
                'id': str(id),
                keys[0]['name']: str(Path('images', image_file)),
                'target': {
                    'instance_id': instance_id,
                    'category_id': category_id,
                    'category_name': category_name,
                    'bbox': bbox,
                },
            })
        save_image_info(dict_image_file, Path(download_dir, 'train'))
        
        # --- save info.json (validation) ---
        if (dataloader.df_instances_validation is not None):
            os.makedirs(Path(download_dir, 'validation', 'images'), exist_ok=True)
            dict_image_file = []
            image_ids = dataloader.df_instances_validation['image_id'].unique()
            file_names = dataloader.df_instances_validation['file_name'].unique()
            instance_ids = dataloader.df_instances_validation.groupby(by=['image_id'], sort=False)['instance_id'].apply(list).values
            bboxes = dataloader.df_instances_validation.groupby(by=['image_id'], sort=False)['bbox'].apply(list).values
            category_ids = dataloader.df_instances_validation.groupby(by=['image_id'], sort=False)['category_id'].apply(list).values
            category_names = dataloader.df_instances_validation.groupby(by=['image_id'], sort=False)['category_name'].apply(list).values
            for id, image_file, instance_id, bbox, category_id, category_name in zip(image_ids, file_names, instance_ids, bboxes, category_ids, category_names):
                # --- copy from COCO directory(src) to AI Dashboard directory(dst)
                if (Path(download_dir, 'train2017', image_file).exists()):
                    src_file = str(Path(download_dir, 'train2017', image_file))
                    dst_file = str(Path(download_dir, 'validation', 'images', image_file))
                    shutil.move(src_file, dst_file)
                
                # --- set image info ---
                dict_image_file.append({
                    'id': str(id),
                    keys[0]['name']: str(Path('images', image_file)),
                    'target': {
                        'instance_id': instance_id,
                        'category_id': category_id,
                        'category_name': category_name,
                        'bbox': bbox,
                    },
                })
            save_image_info(dict_image_file, Path(download_dir, 'validation'))
        
        # --- save info.json (test) ---
        os.makedirs(Path(download_dir, 'test', 'images'), exist_ok=True)
        dict_image_file = []
        image_ids = dataloader.df_instances_test['image_id'].unique()
        file_names = dataloader.df_instances_test['file_name'].unique()
        instance_ids = dataloader.df_instances_test.groupby(by=['image_id'], sort=False)['instance_id'].apply(list).values
        bboxes = dataloader.df_instances_test.groupby(by=['image_id'], sort=False)['bbox'].apply(list).values
        category_ids = dataloader.df_instances_test.groupby(by=['image_id'], sort=False)['category_id'].apply(list).values
        category_names = dataloader.df_instances_test.groupby(by=['image_id'], sort=False)['category_name'].apply(list).values
        for id, image_file, instance_id, bbox, category_id, category_name in zip(image_ids, file_names, instance_ids, bboxes, category_ids, category_names):
            # --- copy from COCO directory(src) to AI Dashboard directory(dst)
            if (Path(download_dir, 'val2017', image_file).exists()):
                src_file = str(Path(download_dir, 'val2017', image_file))
                dst_file = str(Path(download_dir, 'test', 'images', image_file))
                shutil.move(src_file, dst_file)
            
            # --- set image info
            dict_image_file.append({
                'id': str(id),
                keys[0]['name']: str(Path('images', image_file)),
                'target': {
                    'instance_id': instance_id,
                    'category_id': category_id,
                    'category_name': category_name,
                    'bbox': bbox,
                },
            })
        os.makedirs(Path(download_dir, 'test'), exist_ok=True)
        save_image_info(dict_image_file, Path(download_dir, 'test'))
        
    elif (dataset.name == 'PascalVOC2012'):
        # --- Create dataloader object ---
        dataloader = DataLoaderPascalVOC2012(download_dir, validation_split=0.2, download=download)
        
        # --- Create meta data ---
        meta_dir = Path(download_dir, 'meta')
        keys = [{
                    'name': 'img_file',
                    'type': 'image_file',
                }]
        dict_meta = save_meta(meta_dir, 'True', 'object_detection', 'image_data', keys)
    
    elif (dataset.name == 'CaliforniaHousing'):
        # --- Create dataloader object ---
        dataloader = DataLoaderCaliforniaHousing(download_dir)
        
        # --- Create meta data ---
        meta_dir = Path(download_dir, 'meta')
        keys = [{'name': key, 'type': 'number'} for key in dataloader.train_x.keys()]
        dict_meta = save_meta(meta_dir, 'True', 'regression', 'table_data', keys)
        
        # --- save info.json ---
        df_meta = pd.DataFrame(dict_meta)
        save_table_info(df_meta, dataloader.train_x, dataloader.train_y, Path(download_dir, 'train'))
        save_table_info(df_meta, dataloader.validation_x, dataloader.validation_y, Path(download_dir, 'validation'))
        save_table_info(df_meta, dataloader.test_x, dataloader.test_y, Path(download_dir, 'test'))
        
    else:
        # --- Load dataset ---
        meta_dir = Path(dataset.meta_zip.path).parent
        train_dir = Path(dataset.train_zip.path).parent
        
        if (dataset.valid_zip):
            valid_dir = Path(dataset.valid_zip.path).parent
        else:
            valid_dir = None
        
        if (dataset.test_zip):
            test_dir = Path(dataset.test_zip.path).parent
        else:
            test_dir = None
        
        dataloader = DataLoaderCustom()
        flg_verified = dataloader.verify(meta_dir, train_dir, validation_dir=valid_dir, test_dir=test_dir)
        logging.info('-------------------------------------')
        logging.info(f'flg_verified = {flg_verified}')
        logging.info('-------------------------------------')
        if (flg_verified):
            dataloader.load_data(meta_dir, train_dir, validation_dir=valid_dir, test_dir=test_dir, one_hot=False)
    
        logging.info('-------------------------------------')
        logging.info(f'dataloader.verified = {dataloader.verified}')
        logging.info(f'meta_dir = {meta_dir}')
        logging.info(f'train_dir = {train_dir}')
        logging.info(f'valid_dir = {valid_dir}')
        logging.info(f'test_dir = {test_dir}')
        logging.info('-------------------------------------')
        
        # --- Set dataset_type ---
        logging.info('-------------------------------------')
        logging.info(f'dataloader.dataset_type = {dataloader.dataset_type}')
        logging.info('-------------------------------------')
        if ((dataloader.dataset_type == 'img_clf') or (dataloader.dataset_type == 'img_reg')):
            dataset.dataset_type = dataset.DATASET_TYPE_IMAGE
    
    # --- save dataset object to pickle file ---
    with open(Path(download_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataloader, f)
    
    # --- set done status for dataset download ---
    dataset.download_status = dataset.STATUS_DONE
    dataset.save()

    return dataloader

def get_jupyter_nb_url():
    """ Function: get_jupyter_nb_url
     * return Jupyter Notebook URL
    """
    
    return settings.JUPYTER_NB_URL


def get_dataloader_obj(dataset):
    """Get DataLoader object
    
    get DataLoader object from Dataset(models.Model)
    
    Args:
        dataset (models.Model): Dataset model
    
    Returns:
        dataloader_obj (DataLoader): DataLoader object
    
    """
    dataset_dir = Path(settings.MEDIA_ROOT, settings.DATASET_DIR, dataset.project.hash)
    download_dir = Path(dataset_dir, f'dataset_{dataset.id}')
    if (Path(download_dir, 'dataset.pkl').exists()):
        with open(Path(download_dir, 'dataset.pkl'), 'rb') as f:
            dataloader_obj = pickle.load(f)
    else:
        dataloader_obj = None
    
    return dataloader_obj
    
