import json
import logging
import os
import pickle
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from django.conf import settings
from django.http import FileResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse

from app.models import Dataset, MlModel, OperationJob, OperationStep, Project
from machine_learning.lib.utils.utils import JsonEncoder
from views_common import SidebarActiveStatus, get_dataloader_obj, get_jupyter_nb_url, get_version

# Create your views here.

def inference(request):
    """ Function: inference
     * inference top
    """
    def _build_id_to_name(dataset_obj, dataloader_obj):
        """Build mapping from category id to name using available metadata."""
        id_to_name = {}

        dataset_dir = Path(settings.MEDIA_ROOT, settings.DATASET_DIR, dataset_obj.project.hash, f'dataset_{dataset_obj.id}')

        # Prefer explicit class name files recorded in the dataloader.
        candidate_paths = []
        if dataloader_obj:
            for split_key in ('train_dataset', 'validation_dataset', 'test_dataset'):
                split_info = getattr(dataloader_obj, split_key, None)
                if isinstance(split_info, dict):
                    class_name_path = split_info.get('class_name_file_path')
                    if class_name_path:
                        candidate_paths.append(Path(class_name_path))

        # Fallback: common filename alongside dataset contents.
        candidate_paths.append(Path(dataset_dir, 'category_names.txt'))

        for path in candidate_paths:
            if path.exists():
                with open(path, 'r') as f:
                    names = [line.strip() for line in f if line.strip()]
                if names:
                    id_to_name = {idx: name for idx, name in enumerate(names)}
                    break

        # Final fallback: mine info.json for (category_id, category_name) pairs (detection datasets).
        if not id_to_name:
            for split in ('train', 'validation', 'test'):
                info_path = Path(dataset_dir, split, 'info.json')
                if not info_path.exists():
                    continue

                try:
                    with open(info_path, 'r') as f:
                        info_data = json.load(f)
                except Exception:
                    continue

                for entry in info_data:
                    target = entry.get('target')
                    if not isinstance(target, dict):
                        continue

                    class_ids = target.get('class_id')
                    category_names = target.get('category_name')

                    if isinstance(class_ids, list) and isinstance(category_names, list):
                        for cid, cname in zip(class_ids, category_names):
                            try:
                                cid_key = int(cid)
                            except Exception:
                                continue
                            if cname is not None and cid_key not in id_to_name:
                                id_to_name[cid_key] = cname
                    elif class_ids is not None and category_names is not None:
                        try:
                            cid_key = int(class_ids)
                        except Exception:
                            continue
                        if cid_key not in id_to_name:
                            id_to_name[cid_key] = category_names

                if id_to_name:
                    break

        return id_to_name

    def _get_selected_object():
        project_name = request.session.get('inference_view_selected_project', None)
        selected_project = Project.objects.get(name=project_name)
        
        model_name = request.session.get('inference_view_selected_model', None)
        selected_model = MlModel.objects.get(name=model_name, project=selected_project)
        
        return selected_project, selected_model
    
    def _enqueue_inference_job():
        selected_project, selected_model = _get_selected_object()
        if not selected_project or not selected_model:
            raise RuntimeError('Please select project and model')

        existing = OperationJob.objects.filter(
            job_type=OperationJob.JOB_TYPE_INFERENCE_RUN,
            model=selected_model,
            status__in=[OperationJob.STATUS_PENDING, OperationJob.STATUS_RUNNING],
        ).order_by('-created_at').first()
        if existing:
            return existing.id

        job = OperationJob.objects.create(
            job_type=OperationJob.JOB_TYPE_INFERENCE_RUN,
            project=selected_project,
            model=selected_model,
            dataset=selected_model.dataset,
            status=OperationJob.STATUS_RUNNING,
        )
        step_labels = [
            'Validate inputs',
            'Prepare evaluation outputs',
            'Run inference worker',
            'Verify predictions',
            'Done',
        ]
        for idx, label in enumerate(step_labels, start=1):
            OperationStep.objects.create(job=job, order=idx, label=label, status=OperationStep.STATUS_PENDING)
        OperationStep.objects.filter(job=job, order=1).update(status=OperationStep.STATUS_RUNNING)

        manage_py = Path(settings.BASE_DIR, 'manage.py')
        subprocess.Popen(
            [sys.executable, str(manage_py), 'run_operation_job', str(job.id)],
            cwd=str(settings.BASE_DIR),
        )
        return job.id

    # logging.info('-------------------------------------')
    # logging.info(request.method)
    # logging.info(request.POST)
    # logging.info('-------------------------------------')
    if (request.method == 'POST'):
        if ('inference_view_project_dropdown' in request.POST):
            request.session['inference_view_selected_project'] = request.POST.getlist('inference_view_project_dropdown')[0]
                
        elif ('inference_view_model_dropdown' in request.POST):
            request.session['inference_view_selected_model'] = request.POST.getlist('inference_view_model_dropdown')[0]
            
        elif ('inference_view_dataset_dropdown' in request.POST):
            pass
            # (T.B.D)
            #   * dataset dropdown will be selected dataset that user required to inference
            
            # for debug
            # request.session['inference_view_selected_dataset'] = request.POST.getlist('inference_view_dataset_dropdown')[0]
            # curr_project = Project.objects.get(name=request.session['inference_view_selected_project'])
            # curr_dataset = Dataset.objects.get(name=request.session['inference_view_selected_dataset'], project=curr_project)
            
        elif ('inference_run' in request.POST):
            try:
                job_id = _enqueue_inference_job()
            except Exception as exc:
                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({'error': str(exc)}, status=400)
                raise

            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'job_id': job_id})
            return redirect(reverse('inference') + f'?job={job_id}')
        
        elif ('prediction_filter' in request.POST):
            request.session['prediction_filter'] = request.POST.getlist('prediction_filter')[0]
        
        elif ('prediction_data_type' in request.POST):
            request.session['prediction_data_type'] = request.POST.getlist('prediction_data_type')[0]
        
        else:
            logging.warning('Unknown POST command:')
            logging.warning(request.POST)
        
        return redirect('inference')
    else:
        sidebar_status = SidebarActiveStatus()
        sidebar_status.inference = 'active'
        
        project = Project.objects.all().order_by('-id').reverse()
        
        # check for existence of selected project name
        project_name_list = [p.name for p in project]
        selected_project_name = request.session.get('inference_view_selected_project', None)
        
        logging.info('-------------------------------------')
        logging.info(project_name_list)
        logging.info(selected_project_name)
        logging.info('-------------------------------------')
        
        if ((selected_project_name is not None) and (selected_project_name in project_name_list)):
            project_dropdown_selected = Project.objects.get(name=selected_project_name)
        else:
            project_dropdown_selected = None
        
        if (project_dropdown_selected):
            # --- get model list and selected model ---
            model = MlModel.objects.filter(project=project_dropdown_selected).order_by('-id').reverse()
            
            model_name = request.session.get('inference_view_selected_model', None)
            if (model_name in [f.name for f in MlModel.objects.filter(project=project_dropdown_selected)]):
                model_dropdown_selected = MlModel.objects.get(name=model_name, project=project_dropdown_selected)
            else:
                model_dropdown_selected = None
            
            # --- get dataset list and selected dataset (T.B.D) ---
            dataset = Dataset.objects.filter(project=project_dropdown_selected).order_by('-id').reverse()
            if (model_dropdown_selected is not None):
                dataset_dropdown_selected = model_dropdown_selected.dataset
            else:
                dataset_dropdown_selected = None
            
            #
            #dataset_name_list = [d.name for d in Dataset.objects.all().order_by('-id')]
            #selected_dataset_name = request.session.get('inference_view_selected_dataset', None)
            #logging.info('-------------------------------------')
            #logging.info(dataset_name_list)
            #logging.info(selected_dataset_name)
            #logging.info('-------------------------------------')
            #if ((selected_dataset_name is not None) and (selected_dataset_name in dataset_name_list)):
            #    dataset_dropdown_selected = Dataset.objects.get(name=selected_dataset_name, project=project_dropdown_selected)
            #else:
            #    dataset_dropdown_selected = None
            
            
        else:
            model = MlModel.objects.all().order_by('-id').reverse()
            model_dropdown_selected = None
        
            dataset = Dataset.objects.all().order_by('-id').reverse()
            dataset_dropdown_selected = None
        
        # --- Check prediction filter ---
        prediction_filter_selected = request.session.get('prediction_filter', 'All')
        
        # --- Check prediction data type
        prediction_data_type_selected = request.session.get('prediction_data_type', 'Test')
        
        inference_job_id = request.GET.get('job')
        running_job = None
        if not inference_job_id and model_dropdown_selected:
            running_job = OperationJob.objects.filter(
                job_type=OperationJob.JOB_TYPE_INFERENCE_RUN,
                model=model_dropdown_selected,
                status__in=[OperationJob.STATUS_PENDING, OperationJob.STATUS_RUNNING],
            ).order_by('-created_at').first()
            if running_job:
                inference_job_id = str(running_job.id)

        # --- Load DataLoader object and prediction (skip while job is running) ---
        prediction = None
        dataloader_obj = None
        if inference_job_id:
            pass
        elif dataset_dropdown_selected is not None and model_dropdown_selected is not None:
            dataloader_obj = get_dataloader_obj(dataset_dropdown_selected)
            id_to_name = _build_id_to_name(dataset_dropdown_selected, dataloader_obj)

            if dataloader_obj.dataset_type == 'img_det':
                summary_path = Path(
                    model_dropdown_selected.model_dir,
                    'evaluations',
                    f'{prediction_data_type_selected.lower()}_detection_summary.json',
                )
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        prediction_raw = json.load(f)

                    prediction = []
                    for record in prediction_raw:
                        filename = record.get('filename')
                        image_url = None
                        overlay_url = None
                        gt_overlay_url = None

                        if filename:
                            filename_path = Path(filename)
                            if '..' not in filename_path.parts:
                                rel_path = PurePosixPath(
                                    settings.DATASET_DIR,
                                    dataset_dropdown_selected.project.hash,
                                    f'dataset_{dataset_dropdown_selected.id}',
                                    prediction_data_type_selected.lower(),
                                    filename_path.as_posix(),
                                )
                                image_url = f"{settings.MEDIA_URL.rstrip('/')}/{quote(str(rel_path), safe='/')}"

                        overlay_rel = record.get('overlay_relpath')
                        if overlay_rel:
                            try:
                                base_rel = Path(model_dropdown_selected.model_dir).relative_to(settings.MEDIA_ROOT)
                                overlay_rel_path = PurePosixPath(base_rel, overlay_rel)
                                overlay_url = f"{settings.MEDIA_URL.rstrip('/')}/{quote(str(overlay_rel_path), safe='/')}"
                            except Exception:
                                overlay_url = None

                        gt_overlay_rel = record.get('gt_overlay_relpath')
                        if gt_overlay_rel:
                            try:
                                base_rel = Path(model_dropdown_selected.model_dir).relative_to(settings.MEDIA_ROOT)
                                gt_overlay_rel_path = PurePosixPath(base_rel, gt_overlay_rel)
                                gt_overlay_url = f"{settings.MEDIA_URL.rstrip('/')}/{quote(str(gt_overlay_rel_path), safe='/')}"
                            except Exception:
                                gt_overlay_url = None

                        record_with_urls = dict(record)
                        record_with_urls['image_url'] = image_url
                        record_with_urls['overlay_url'] = overlay_url
                        record_with_urls['gt_overlay_url'] = gt_overlay_url
                        prediction.append(record_with_urls)
            else:
                prediction_json = Path(model_dropdown_selected.model_dir, 'evaluations', f'{prediction_data_type_selected.lower()}_prediction.json')
                if prediction_json.exists():
                    with open(prediction_json, 'r') as f:
                        prediction_raw = json.load(f)

                    prediction = []
                    for record in prediction_raw:
                        filename = record.get('filename')
                        thumbnail_url = None

                        if filename:
                            filename_path = Path(filename)
                            if '..' not in filename_path.parts:
                                rel_path = PurePosixPath(
                                    settings.DATASET_DIR,
                                    dataset_dropdown_selected.project.hash,
                                    f'dataset_{dataset_dropdown_selected.id}',
                                    prediction_data_type_selected.lower(),
                                    filename_path.as_posix(),
                                )
                                thumbnail_url = f"{settings.MEDIA_URL.rstrip('/')}/{quote(str(rel_path), safe='/')}"

                        record_with_thumbnail = dict(record)
                        record_with_thumbnail['thumbnail_url'] = thumbnail_url

                        pred_id = record.get('prediction')
                        tgt_id = record.get('target')
                        record_with_thumbnail['prediction_name'] = id_to_name.get(pred_id)
                        record_with_thumbnail['target_name'] = id_to_name.get(tgt_id)
                        prediction.append(record_with_thumbnail)
        
        
        context = {
            'project': project,
            'model': model,
            'dataset': dataset,
            'sidebar_status': sidebar_status,
            'text': get_version(),
            'jupyter_nb_url': get_jupyter_nb_url(),
            'project_dropdown_selected': project_dropdown_selected,
            'model_dropdown_selected': model_dropdown_selected,
            'dataset_dropdown_selected': dataset_dropdown_selected,
            'prediction': prediction,
            'prediction_filter_selected': prediction_filter_selected,
            'prediction_data_type_selected': prediction_data_type_selected,
            'dataloader_obj': dataloader_obj,
            'job_id': inference_job_id,
        }
        return render(request, 'inference.html', context)

def download_prediction(request):
    """ Function: download_prediction
     * download prediction
    """
    
    # --- get selected project and model ---
    selected_project_name = request.session.get('inference_view_selected_project', None)
    selected_model_name = request.session.get('inference_view_selected_model', None)
    
    selected_project = Project.objects.get(name=selected_project_name)
    selected_model = MlModel.objects.get(name=selected_model_name, project=selected_project)
    
    # --- get prediction ---
    prediction_data_type_selected = request.session.get('prediction_data_type', 'Test')
    prediction_csv = Path(selected_model.model_dir, f'{prediction_data_type_selected.lower()}_prediction.csv')
    
    return FileResponse(open(prediction_csv, "rb"), as_attachment=True, filename=prediction_csv.name)
    
