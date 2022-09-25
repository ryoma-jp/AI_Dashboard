import os
import pickle
import logging
import json
import cv2
import shutil

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings

from app.models import Project, Dataset
from app.forms import DatasetForm

from views_common import SidebarActiveStatus, get_version, load_dataset, get_jupyter_nb_url

# Create your views here.

def _save_image_files(images, image_shape, labels, output_dir, name='images'):
    """Save Image Files
    
    Convert image data to image file and save to <dataset_dir>/<name>
    Internal function of views_dataset
    
    Args:
        images: Image list
        image_shape: image shape (tuple)
        labels: classification label (ground truth, one_hot)
        output_dir: output directory
        name: data name
    
    Return:
        None
    """
    
    # --- create output directory ---
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    
    # --- save image files ---
    dict_image_file = {
        'id': [],
        'file': [],
        'class_id': [],
    }
    for i, (image, label) in enumerate(zip(images, labels)):
        image_file = os.path.join(name, f'{i:08}.png')
        image = image.reshape(image_shape)
        cv2.imwrite(os.path.join(output_dir, image_file), image)
        
        dict_image_file['id'].append(i)
        dict_image_file['file'].append(image_file)
        # dict_image_file['class_id'].append(int(np.argmax(label)))
        dict_image_file['class_id'].append(int(label))
    
    # --- save image files information to json file ---
    with open(os.path.join(output_dir, f'info_{name}.json'), 'w') as f:
        json.dump(dict_image_file, f, ensure_ascii=False, indent=4)
    
    return None
    

def dataset(request):
    """ Function: dataset
     * dataset top
    """
    
    # --- reset dataset detail parameters ---
    if ('dropdown_dataset_info' in request.session.keys()):
        del request.session['dropdown_dataset_info']
        del request.session['selected_dataset_type']
    
    if (request.method == 'POST'):
        if ('dataset_view_dropdown' in request.POST):
            request.session['dataset_view_dropdown_selected_project'] = request.POST.getlist('dataset_view_dropdown')[0]
        
        elif ('dataset_view_upload' in request.POST):
            form_custom_dataset = DatasetForm(request.POST, request.FILES)
            if (form_custom_dataset.is_valid()):
                # --- get related project ---
                project_name = request.session.get('dataset_view_dropdown_selected_project', None)
                project = Project.objects.get(name=project_name)
                
                # --- save model ---
                dataset = Dataset.objects.create(
                              name=form_custom_dataset.cleaned_data.get('name'),
                              project=project,
                              train_zip=form_custom_dataset.cleaned_data.get('train_zip'),
                              valid_zip=form_custom_dataset.cleaned_data.get('valid_zip'),
                              test_zip=form_custom_dataset.cleaned_data.get('test_zip'),
                              uploaded_at=form_custom_dataset.cleaned_data.get('uploaded_at'),
                              download_status=Dataset.STATUS_NONE,
                              image_gallery_status=Dataset.STATUS_NONE,
                          )
        
                # --- unzip ---
                train_dir = os.path.dirname(dataset.train_zip.path)
                valid_dir = os.path.dirname(dataset.valid_zip.path)
                test_dir = os.path.dirname(dataset.test_zip.path)
                
                shutil.unpack_archive(dataset.train_zip.path, train_dir)
                shutil.unpack_archive(dataset.valid_zip.path, valid_dir)
                shutil.unpack_archive(dataset.test_zip.path, test_dir)
                
                # --- preparing ---
                load_dataset(dataset)
                
        return redirect('dataset')
        
    else:
        project = Project.objects.all().order_by('-id').reverse()
        dataset = Dataset.objects.all().order_by('-id').reverse()
        sidebar_status = SidebarActiveStatus()
        sidebar_status.dataset = 'active'
        
        project_name = request.session.get('dataset_view_dropdown_selected_project', None)
        if (project_name is not None):
            dropdown_selected_project = Project.objects.get(name=project_name)
        else:
            dropdown_selected_project = None
        
        form_custom_dataset = DatasetForm()
        
        # logging.info('-------------------------------------')
        # logging.info(project_name)
        # logging.info(dropdown_selected_project)
        # logging.info('-------------------------------------')
        context = {
            'project': project,
            'dataset': dataset,
            'sidebar_status': sidebar_status,
            'text': get_version(),
            'jupyter_nb_url': get_jupyter_nb_url(),
            'dropdown_selected_project': dropdown_selected_project,
            'form_custom_dataset': form_custom_dataset,
        }
        return render(request, 'dataset.html', context)

def dataset_detail(request, project_id, dataset_id):
    """ Function: dataset_detail
     * display dataset details(images, distribution, etc)
    """
    
    def _get_dataloader_obj(dataset):
        dataset_dir = os.path.join(settings.MEDIA_ROOT, settings.DATASET_DIR, dataset.project.hash)
        download_dir = os.path.join(dataset_dir, f'dataset_{dataset.id}')
        if (os.path.exists(os.path.join(download_dir, 'dataset.pkl'))):
            download_button_state = "disabled"
            with open(os.path.join(download_dir, 'dataset.pkl'), 'rb') as f:
                dataloader_obj = pickle.load(f)
        else:
            download_button_state = ""
            dataloader_obj = None
        
        return dataloader_obj, download_button_state, download_dir
    
    
    # logging.info('-------------------------------------')
    # logging.info(request)
    # logging.info(request.method)
    # logging.info('-------------------------------------')
    
    dataset_info = [
        'Images',
        'Statistics',
    ]
    
    project = get_object_or_404(Project, pk=project_id)
    dataset = get_object_or_404(Dataset, pk=dataset_id, project=project)
    
    if (request.method == 'POST'):
        logging.info('-------------------------------------')
        logging.info(request.POST)
        logging.info(request.POST.keys())
        logging.info('-------------------------------------')
        
        if ('dataset_download' in request.POST.keys()):
            # --- dataset download ---
            dataset.download_status = dataset.STATUS_PREPARING
            dataset.save()
            
            context = {
                'text': get_version(),
                'jupyter_nb_url': get_jupyter_nb_url(),
                'project_id': project.id,
                'dataset_id': dataset.id,
                'dataset_name': dataset.name,
                'download_status': dataset.download_status,
            }
            return render(request, 'dataset_detail.html', context)
        
        if ('dropdown_dataset_info' in request.POST.keys()):
            selected_dataset_info = request.POST['dropdown_dataset_info']
            request.session['dropdown_dataset_info'] = selected_dataset_info
            
            if ((selected_dataset_info == 'Images') and (dataset.image_gallery_status == dataset.STATUS_NONE)):
                dataset.image_gallery_status = dataset.STATUS_PREPARING
                dataset.save()
            
                dataloader_obj, download_button_state, download_dir = _get_dataloader_obj(dataset)
                context = {
                    'text': get_version(),
                    'jupyter_nb_url': get_jupyter_nb_url(),
                    'project_id': project.id,
                    'dataset_id': dataset.id,
                    'dataset_name': dataset.name,
                    'dataloader_obj': dataloader_obj,
                    'download_status': dataset.download_status,
                    'download_button_state': download_button_state,
                    'dataset_info': dataset_info,
                    'selected_dataset_info': selected_dataset_info,
                    'image_gallery_status': dataset.image_gallery_status,
                }
                return render(request, 'dataset_detail.html', context)
            
        if ('image_gallery_key' in request.POST.keys()):
            selected_dataset_type = request.POST['image_gallery_key']
            request.session['selected_dataset_type'] = selected_dataset_type
            request.session['image_gallery_page_now'] = 1
        
        if ('select_page' in request.POST.keys()):
            selected_page = int(request.POST['select_page'])
            request.session['image_gallery_page_now'] = selected_page
            
    # --- check dataset download ---
    if (dataset.download_status == dataset.STATUS_PREPARING):
        load_dataset(dataset)
        
    # --- check download directory ---
    if (dataset.download_status == dataset.STATUS_DONE):
        dataloader_obj, download_button_state, download_dir = _get_dataloader_obj(dataset)
        
        # --- preparing to display images ---
        if (dataset.image_gallery_status == dataset.STATUS_PREPARING):
            dataset.image_gallery_status = dataset.STATUS_PROCESSING
            dataset.save()
            
            if (dataloader_obj.train_images is not None):
                _save_image_files(dataloader_obj.train_images, dataloader_obj.train_images.shape[1:],
                                 dataloader_obj.train_labels, download_dir, name='train_images')
            if (dataloader_obj.validation_images is not None):
                _save_image_files(dataloader_obj.validation_images, dataloader_obj.validation_images.shape[1:],
                                 dataloader_obj.validation_labels, download_dir, name='validation_images')
            if (dataloader_obj.test_images is not None):
                _save_image_files(dataloader_obj.test_images, dataloader_obj.test_images.shape[1:],
                                 dataloader_obj.test_labels, download_dir, name='test_images')
            
            dataset.image_gallery_status = dataset.STATUS_DONE
            dataset.save()
            
        selected_dataset_info = request.session.get('dropdown_dataset_info', None)
        selected_dataset_type = request.session.get('selected_dataset_type', None)
        
        # --- set keys ---
        image_gallery_keys = []
        if (dataloader_obj.train_images is not None):
            image_gallery_keys.append('train_images')
        if (dataloader_obj.validation_images is not None):
            image_gallery_keys.append('validation_images')
        if (dataloader_obj.test_images is not None):
            image_gallery_keys.append('test_images')
        
        # --- set image data file ---
        image_gallery_data = []
        if (selected_dataset_type is not None):
            images_page_now = request.session.get('image_gallery_page_now', 1)
            images_per_page = 50
            
            json_file = os.path.join(download_dir, f'info_{selected_dataset_type}.json')
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            images_page_max = len(json_data['id']) // images_per_page
            images_page_list = [x for x in range(1, images_page_max+1)]
            
            for i in range((images_page_now-1)*images_per_page, ((images_page_now-1)*images_per_page)+images_per_page):
                image_gallery_data.append({
                    'id': json_data['id'][i],
                    'file': os.path.join(settings.MEDIA_URL,
                                         settings.DATASET_DIR,
                                         dataset.project.hash,
                                         f'dataset_{dataset.id}',
                                         json_data['file'][i]),
                    'class_id': json_data['class_id'][i],
                })
        else:
            images_page_now = 1
            images_page_max = 1
            images_page_list = []
            
        
        context = {
            'text': get_version(),
            'jupyter_nb_url': get_jupyter_nb_url(),
            'dataset_name': dataset.name,
            'dataloader_obj': dataloader_obj,
            'download_status': dataset.download_status,
            'download_button_state': download_button_state,
            'dataset_info': dataset_info,
            'selected_dataset_info': selected_dataset_info,
            'image_gallery_status': dataset.image_gallery_status,
            'image_gallery_keys': image_gallery_keys,
            'image_gallery_selected_item': selected_dataset_type,
            'image_gallery_data': image_gallery_data,
            'images_page_now': images_page_now,
            'images_page_max': images_page_max,
            'images_page_list': images_page_list,
        }
        return render(request, 'dataset_detail.html', context)
    else:
        context = {
            'text': get_version(),
            'jupyter_nb_url': get_jupyter_nb_url(),
            'dataset_name': dataset.name,
            'download_status': dataset.download_status,
        }
        return render(request, 'dataset_detail.html', context)

