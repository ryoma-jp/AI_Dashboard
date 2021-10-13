import logging
import threading
import concurrent.futures

from django.shortcuts import render
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from app.models import DatasetFile, DatasetSelection
from app.forms import DatasetFileForm, DatasetSelectionForm

from .machine_learning.lib.trainer.trainer import Trainer

# --- executor for machine learning
ml_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Create your views here.

""" Function: index
 * show main view
"""
def index(request):
    def _proc_dataset_selection(request, model):
        if ('selection' in request.POST):
            if (model is not None):
                form = DatasetSelectionForm(request.POST, instance=model)
            else:
                form = DatasetSelectionForm(request.POST)
            
            if (form.is_valid()):
                logging.debug('form is valid')
                form.save()
        else:
            if (model is not None):
                form = DatasetSelectionForm(instance=model)
            else:
                form = DatasetSelectionForm()
        
        return form
    
    def _proc_dataset_upload(request):
        dataset_file_form = DatasetFileForm()
        logging.debug('dataset_file_form.fields: ')
        logging.debug(dataset_file_form.fields)
        if any((key in dataset_file_form.fields.keys()) for key in request.FILES.keys()):
            flg_update_dataset_file = True
        else:
            flg_update_dataset_file = False
        
        logging.debug('flg_update_dataset_file: ')
        logging.debug(flg_update_dataset_file)
        if (flg_update_dataset_file):
            files = request.FILES.keys()
            logging.debug('files: ')
            logging.debug(files)
            
            try:
                file = DatasetFile.objects.get(id=1)
            except ObjectDoesNotExist:
                file = DatasetFile()
            
            if ('x_train' in files):
                dataset_file = request.FILES.getlist('x_train', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.x_train = dataset_file
        
            if ('y_train' in files):
                dataset_file = request.FILES.getlist('y_train', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.y_train = dataset_file
        
            if ('x_valid' in files):
                dataset_file = request.FILES.getlist('x_valid', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.x_valid = dataset_file
        
            if ('y_valid' in files):
                dataset_file = request.FILES.getlist('y_valid', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.y_valid = dataset_file
        
            if ('x_test' in files):
                dataset_file = request.FILES.getlist('x_test', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.x_test = dataset_file
        
            if ('y_test' in files):
                dataset_file = request.FILES.getlist('y_test', False)[0]
                logging.debug('dataset_file: ')
                logging.debug(dataset_file)
                file.y_test = dataset_file
            
            file.save()
        
        return dataset_file_form
    
    def _training_run(request):
        def __thread_func(interval=1):
            thread_id = threading.get_ident()
            Trainer.Counter(interval)
            
            return
        
        logging.debug('training_run: ')
        logging.debug(request.POST.keys())
        if ('training_run' in request.POST.keys()):
            dataset_selection = DatasetSelection.objects.all()
            if (len(dataset_selection) > 0):
                logging.debug(dataset_selection[0].selection)
                interval = 1
                logging.debug(ml_executor._work_queue.empty())
                logging.debug(ml_executor._threads)
                if (ml_executor._work_queue.empty()):
                    ml_executor.submit(__thread_func, interval)
                else:
                    logging.debug('ML thread queue is not empty')
        
        return
    
#    logging.debug('request: ')
#    logging.debug(request.__dict__)
#
#    logging.debug('DatasetSelectionForm: ')
#    logging.debug(DatasetSelectionForm.__dict__)
#
#    logging.debug('DatasetSelection: ')
#    logging.debug(DatasetSelection.__dict__)

    try:
        dataset_selection = DatasetSelection.objects.get(pk=1)
    except ObjectDoesNotExist:
        dataset_selection = None
    logging.debug('dataset_selection: ')
    logging.debug(dataset_selection)
    
    if (request.method == 'POST'):
        logging.debug('request.POST: ')
        logging.debug(request.POST)
        logging.debug('request.FILES: ')
        logging.debug(request.FILES)
        
        # --- データセット選択プルダウン ---
        dataset_selection_form = _proc_dataset_selection(request, dataset_selection)
        dataset_selection = DatasetSelection.objects.get(pk=1)
        
        # --- データセットアップロードフォーム ---
        dataset_file_form = _proc_dataset_upload(request)
        
        # --- 学習実行 ---
        logging.debug('dataset_selection_form')
        logging.debug(dataset_selection_form)
        _training_run(request)
        
    else:
        dataset_file_form = DatasetFileForm()
        if (dataset_selection is not None):
            dataset_selection_form = DatasetSelectionForm(instance=dataset_selection)
        else:
            dataset_selection_form = DatasetSelectionForm()
    
    if (settings.DEBUG):
        text = 'Debug mode'
    else:
        text = '[T.B.D] VerX.XX'
    
    dataset_file = DatasetFile.objects.all()
    if (dataset_selection is None):
        dataset_selection = DatasetSelection()
    system_info = Trainer.GetSystemInfo()
    return render(request, 'index.html', {
               'dataset_file': dataset_file,
               'dataset_selection': dataset_selection,
               'dataset_file_form': dataset_file_form,
               'dataset_selection_form': dataset_selection_form,
               'text': text,
               'system_info': system_info})

