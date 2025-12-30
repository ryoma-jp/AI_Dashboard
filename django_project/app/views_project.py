import json
import hashlib
import pickle
import shutil
import psutil
import signal
import sys
import subprocess
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from app.forms import MlModelForm, ProjectForm
from app.models import AIModelSDK, Dataset, MlModel, OperationJob, OperationStep, Project
from views_common import get_jupyter_nb_url, get_version, load_dataset


#---------------------------------------
# Utility functions
#---------------------------------------

def _create_project_hash(project):
    """create project hash"""
    return hashlib.sha256(f"{project.id:08}".encode()).hexdigest()


def _create_model_hash(project, model):
    """create model hash"""
    return hashlib.sha256(f"{project.id:08}{model.id:08}".encode()).hexdigest()


def _model_delete(request, model):
    """Delete model and its resources"""

    model_dir = Path(model.model_dir)
    if model_dir.exists():
        shutil.rmtree(model_dir)

    if model.tensorboard_pid is not None and psutil.pid_exists(model.tensorboard_pid):
        proc = psutil.Process(model.tensorboard_pid)
        children = proc.children(recursive=True)
        children.append(proc)
        for child in children:
            try:
                child.send_signal(signal.SIGTERM)
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(children, timeout=3)

    if request.session.get('training_view_selected_model') == model.name:
        del request.session['training_view_selected_model']
        request.session.modified = True
    if request.session.get('inference_view_selected_model') == model.name:
        del request.session['inference_view_selected_model']
        request.session.modified = True

    model.delete()


def _project_delete(request, project):
    """Delete project and its resources"""

    for mdl in MlModel.objects.filter(project=project):
        _model_delete(request, mdl)

    project_dir = Path(settings.MEDIA_ROOT, settings.MODEL_DIR, project.hash)
    if project_dir.exists():
        shutil.rmtree(project_dir)

    dataset_dir = Path(settings.MEDIA_ROOT, settings.DATASET_DIR, project.hash)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    ai_model_sdk_dir = Path(settings.MEDIA_ROOT, settings.AI_MODEL_SDK_DIR, 'user_custom_sdk', project.hash)
    if ai_model_sdk_dir.exists():
        shutil.rmtree(ai_model_sdk_dir)

    if request.session.get('training_view_selected_project') == project.name:
        del request.session['training_view_selected_project']
        request.session.modified = True
    if request.session.get('inference_view_selected_project') == project.name:
        del request.session['inference_view_selected_project']
        request.session.modified = True

    project.delete()


def project_new(request):
    """ Function: project_new
     * new project
    """
    if request.method == 'POST':
        form = ProjectForm(request.POST)
        if form.is_valid():
            project = form.save(commit=True)
            project.hash = _create_project_hash(project)
            project.save()

            job = OperationJob.objects.create(
                job_type=OperationJob.JOB_TYPE_PROJECT_CREATE,
                project=project,
                status=OperationJob.STATUS_RUNNING,
            )
            step_labels = [
                'Save project record',
                'Create default datasets',
                'Register sample SDKs',
                'Create project directories',
                'Done',
            ]
            for idx, label in enumerate(step_labels, start=1):
                OperationStep.objects.create(job=job, order=idx, label=label, status=OperationStep.STATUS_PENDING)
            OperationStep.objects.filter(job=job, order=1).update(status=OperationStep.STATUS_DONE)
            OperationStep.objects.filter(job=job, order=2).update(status=OperationStep.STATUS_RUNNING)

            manage_py = Path(settings.BASE_DIR, 'manage.py')
            subprocess.Popen(
                [sys.executable, str(manage_py), 'run_operation_job', str(job.id)],
                cwd=str(settings.BASE_DIR),
            )
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'job_id': job.id})
            return redirect(reverse('project_new') + f'?job={job.id}')
    else:
        form = ProjectForm()

    context = {
        'form': form,
        'job_id': request.GET.get('job', None),
        'text': get_version(),
        'jupyter_nb_url': get_jupyter_nb_url(),
    }
    return render(request, 'project_new.html', context)


def project_edit(request, project_id):
    """ Function: project_edit
     * edit project
    """
    project = get_object_or_404(Project, pk=project_id)
    if request.method == 'POST':
        if 'project_apply' in request.POST:
            form = ProjectForm(request.POST)
            if form.is_valid():
                project.name = form.cleaned_data.get('name')
                project.description = form.cleaned_data.get('description')
                project.save()

                if 'training_view_selected_project' in request.session.keys():
                    del request.session['training_view_selected_project']
                    request.session.modified = True

                return redirect('index')

        elif 'project_delete' in request.POST:
            _project_delete(request, project)
            return redirect('index')
    else:
        initial_dict = dict(name=project.name, description=project.description)
        form = ProjectForm(initial=initial_dict)

    context = {
        'form': form,
        'text': get_version(),
        'jupyter_nb_url': get_jupyter_nb_url(),
    }
    return render(request, 'project_edit.html', context)


def model_new(request, project_id):
    """ Function: model_new
     * new model (job-based)
    """

    project = get_object_or_404(Project, pk=project_id)

    if request.method == 'POST':
        form = MlModelForm(request.POST)
        if form.is_valid():
            selected_model_list = request.POST.getlist('model_new_model_dropdown_submit')
            selected_dataset_list = request.POST.getlist('model_new_dataset_dropdown_submit')
            if not selected_model_list or selected_model_list[0].strip() == '' or not selected_dataset_list or selected_dataset_list[0].strip() == '':
                context = {
                    'model_new_model_selected': None,
                    'model_new_dataset_selected': None,
                    'dataset': Dataset.objects.filter(project=project).order_by('-id').reverse(),
                    'model': AIModelSDK.objects.filter(project=project).order_by('-id').reverse(),
                    'form': form,
                    'text': get_version(),
                    'jupyter_nb_url': get_jupyter_nb_url(),
                    'error_message': 'Please select Model and Dataset.',
                    'job_id': None,
                }
                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({'error': context['error_message']}, status=400)
                return render(request, 'model_new.html', context)

            payload = {
                'project_id': project.id,
                'model_name': form.cleaned_data.get('name'),
                'description': form.cleaned_data.get('description') or '',
                'dataset_name': selected_dataset_list[0],
                'sdk_name': selected_model_list[0],
            }

            job = OperationJob.objects.create(
                job_type=OperationJob.JOB_TYPE_MODEL_CREATE,
                project=project,
                status=OperationJob.STATUS_RUNNING,
                payload=payload,
            )
            step_labels = [
                'Validate inputs',
                'Create model record + hash',
                'Create model/env directories',
                'Prepare dataset (download/verify/build dataset.pkl)',
                'Write config.json + FIFOs',
                'Save model metadata',
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
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'job_id': job.id})
            return redirect(reverse('model_new', args=[project_id]) + f'?job={job.id}')
    else:
        form = MlModelForm()

    model_new_dataset_selected = None
    dataset = Dataset.objects.filter(project=project).order_by('-id').reverse()

    model_new_model_selected = None

    ai_model_sdk = AIModelSDK.objects.filter(project=project).order_by('-id').reverse()

    context = {
        'model_new_model_selected': model_new_model_selected,
        'model_new_dataset_selected': model_new_dataset_selected,
        'dataset': dataset,
        'model': ai_model_sdk,
        'form': form,
        'error_message': None,
        'job_id': request.GET.get('job', None),
        'text': get_version(),
        'jupyter_nb_url': get_jupyter_nb_url(),
    }
    return render(request, 'model_new.html', context)


def model_edit(request, project_id, model_id):
    """ Function: model_edit
     * edit model
    """
    project = get_object_or_404(Project, pk=project_id)
    model = get_object_or_404(MlModel, pk=model_id, project=project)

    if request.method == 'POST':
        if 'model_apply' in request.POST:
            form = MlModelForm(request.POST)
            if form.is_valid():
                model.name = form.cleaned_data.get('name')
                model.description = form.cleaned_data.get('description')

                selected_dataset = request.POST.getlist('model_edit_dataset_dropdown_selected_submit')[0]
                model.dataset = get_object_or_404(Dataset.objects.filter(project=project, name=selected_dataset))

                with open(Path(model.model_dir, 'config.json'), 'r') as f:
                    config_data = json.load(f)
                dataset_dir = config_data['dataset']['dataset_dir']['value']

                dataset = load_dataset(model.dataset)
                model.dataset_pickle = Path(dataset_dir, 'dataset.pkl')
                with open(model.dataset_pickle, 'wb') as f:
                    pickle.dump(dataset, f)

                model.save()

                if 'training_view_selected_model' in request.session.keys():
                    del request.session['training_view_selected_model']
                    request.session.modified = True

                return redirect('index')

        if 'model_delete' in request.POST:
            _model_delete(request, model)
            return redirect('index')

    else:
        initial_dict = dict(name=model.name, description=model.description)
        form = MlModelForm(initial=initial_dict)

    model_edit_dropdown_selected = model.dataset
    dataset = Dataset.objects.filter(project=project).order_by('-id').reverse()

    context = {
        'model_edit_dropdown_selected': model_edit_dropdown_selected,
        'dataset': dataset,
        'form': form,
        'text': get_version(),
        'jupyter_nb_url': get_jupyter_nb_url(),
    }
    return render(request, 'model_edit.html', context)
