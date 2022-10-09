import os
import fcntl
import logging
import json
import pickle

from pathlib import Path

from django.conf import settings

from app.models import Project, MlModel

from machine_learning.lib.data_loader.data_loader import DataLoaderCIFAR10, DataLoaderMNIST, DataLoaderCustom

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
        dataloader = DataLoaderMNIST(download_dir, validation_split=0.2, one_hot=False, download=download)
    elif (dataset.name == 'CIFAR-10'):
        dataloader = DataLoaderCIFAR10(download_dir, validation_split=0.2, one_hot=False, download=download)
    else:
        train_dir = Path(dataset.train_zip.path).parent
        valid_dir = Path(dataset.valid_zip.path).parent
        test_dir  = Path(dataset.test_zip.path).parent
        dataloader = DataLoaderCustom(train_dir, test_dir, validation_dir=valid_dir, one_hot=False)
    
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


