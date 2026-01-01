import os
import json
import pickle
import logging
import uuid
import shutil
import subprocess
import sys

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from app.models import OperationJob, OperationStep, Dataset, AIModelSDK, MlModel, Project
from app.views_project import _create_model_hash
from views_common import load_dataset


class Command(BaseCommand):
    help = 'Run an OperationJob (PROJECT_CREATE / MODEL_CREATE) and update OperationStep statuses.'

    def add_arguments(self, parser):
        parser.add_argument('job_id', type=int)

    def _set_step(self, job, order, status, detail=''):
        OperationStep.objects.filter(job=job, order=order).update(status=status, detail=detail)

    def _fail(self, job, exc, order=None):
        logging.exception('Operation job failed')
        if order is not None:
            self._set_step(job, order, OperationStep.STATUS_ERROR, detail=str(exc))
        job.status = OperationJob.STATUS_ERROR
        job.message = str(exc)
        job.save()

    def _finish(self, job, message='Done'):
        job.status = OperationJob.STATUS_DONE
        job.message = message
        job.save()

    def _ensure_dir(self, path: Path):
        os.makedirs(path, exist_ok=True)

    def _run_project_create(self, job: OperationJob):
        project = job.project
        if project is None:
            raise RuntimeError('PROJECT_CREATE requires project')

        # Step 2: Create default datasets
        self._set_step(job, 2, OperationStep.STATUS_RUNNING)
        defaults = [
            ('MNIST', Dataset.DATASET_TYPE_IMAGE),
            ('CIFAR-10', Dataset.DATASET_TYPE_IMAGE),
            ('COCO2017', Dataset.DATASET_TYPE_IMAGE),
            ('COCO2017 Light', Dataset.DATASET_TYPE_IMAGE),
            ('CaliforniaHousing', Dataset.DATASET_TYPE_TABLE),
        ]
        for name, dataset_type in defaults:
            Dataset.objects.get_or_create(name=name, project=project, defaults={'dataset_type': dataset_type})
        self._set_step(job, 2, OperationStep.STATUS_DONE)

        # Step 3: Register sample SDKs
        self._set_step(job, 3, OperationStep.STATUS_RUNNING)
        sample_sdk_path = Path(settings.MEDIA_ROOT, settings.AI_MODEL_SDK_DIR, 'sample_sdk')
        sample_sdk_list = [
            ('SimpleCNN for MNIST', 'SimpleCNN_for_MNIST', 'SimpleCNN'),
            ('SimpleCNN for CIFAR-10', 'SimpleCNN_for_CIFAR-10', 'SimpleCNN'),
            ('LightGBM for CaliforniaHousing', 'LightGBM_for_CaliforniaHousing', 'LightGBM'),
            ('YOLOv3 for COCO2017', 'YOLOv3_for_COCO2017', 'YOLOv3'),
        ]
        for name, path, model_type in sample_sdk_list:
            AIModelSDK.objects.get_or_create(
                name=name,
                project=project,
                defaults={
                    'model_type': model_type,
                    'ai_model_sdk_dir': Path(sample_sdk_path, path),
                    'ai_model_sdk_dir_offset': Path('sample_sdk', path),
                },
            )
        self._set_step(job, 3, OperationStep.STATUS_DONE)

        # Step 4: Create directories
        self._set_step(job, 4, OperationStep.STATUS_RUNNING)
        self._ensure_dir(Path(settings.MEDIA_ROOT, settings.MODEL_DIR, project.hash))
        self._ensure_dir(Path(settings.MEDIA_ROOT, settings.DATASET_DIR, project.hash))
        self._ensure_dir(Path(settings.MEDIA_ROOT, settings.AI_MODEL_SDK_DIR, 'user_custom_sdk', project.hash))
        self._set_step(job, 4, OperationStep.STATUS_DONE)

        # Step 5: Done
        self._set_step(job, 5, OperationStep.STATUS_DONE)
        self._finish(job, message='Project created')

    def _run_model_create(self, job: OperationJob):
        payload = job.payload or {}

        # Step 1: Validate inputs
        self._set_step(job, 1, OperationStep.STATUS_RUNNING)
        project_id = payload.get('project_id') or (job.project.id if job.project else None)
        model_name = payload.get('model_name')
        description = payload.get('description', '')
        dataset_name = payload.get('dataset_name')
        sdk_name = payload.get('sdk_name')

        if not project_id or not model_name or not dataset_name or not sdk_name:
            raise RuntimeError('MODEL_CREATE payload is missing required fields')

        project = Project.objects.get(id=project_id)
        dataset = Dataset.objects.get(project=project, name=dataset_name)
        ai_model_sdk = AIModelSDK.objects.get(project=project, name=sdk_name)
        self._set_step(job, 1, OperationStep.STATUS_DONE)

        # Step 2: Create model record + hash
        self._set_step(job, 2, OperationStep.STATUS_RUNNING)
        pending_dir = Path(settings.MEDIA_ROOT, settings.MODEL_DIR, project.hash, f'pending_{uuid.uuid4().hex}')
        self._ensure_dir(pending_dir)
        model = MlModel.objects.create(
            name=model_name,
            description=description,
            project=project,
            dataset=dataset,
            ai_model_sdk=ai_model_sdk,
            status='CREATING',
            model_dir=str(pending_dir),
            hash='',
        )
        model.hash = _create_model_hash(project, model)
        model_dir = Path(settings.MEDIA_ROOT, settings.MODEL_DIR, project.hash, model.hash)
        self._set_step(job, 2, OperationStep.STATUS_DONE)

        # Step 3: Create model/env directories
        self._set_step(job, 3, OperationStep.STATUS_RUNNING)
        self._ensure_dir(model_dir)
        env_dir = Path(settings.ENV_DIR, project.hash, model.hash)
        self._ensure_dir(env_dir)
        dataset_dir = Path(settings.MEDIA_ROOT, settings.DATASET_DIR, project.hash)
        self._ensure_dir(dataset_dir)
        try:
            shutil.rmtree(pending_dir)
        except Exception:
            pass
        model.model_dir = str(model_dir)
        model.save()
        job.model = model
        job.save()
        self._set_step(job, 3, OperationStep.STATUS_DONE)

        # Step 4: Prepare dataset (heavy)
        self._set_step(job, 4, OperationStep.STATUS_RUNNING)
        dataloader = load_dataset(dataset)
        self._set_step(job, 4, OperationStep.STATUS_DONE)

        # Step 5: Write config + FIFOs
        self._set_step(job, 5, OperationStep.STATUS_RUNNING)
        if dataset.name == 'MNIST':
            config_file = 'config_mnist.json'
        elif dataset.name == 'CIFAR-10':
            config_file = 'config_cifar10.json'
        elif dataset.name == 'COCO2017':
            config_file = 'config_coco2017.json'
        elif dataset.name == 'COCO2017 Light':
            config_file = 'config_coco2017_light.json'
        elif dataset.name == 'CaliforniaHousing':
            config_file = 'config_california_housing.json'
        else:
            config_file = 'config_blank.json'

        with open(Path(settings.MEDIA_ROOT, settings.CONFIG_DIR, config_file), 'r') as f:
            dict_config = json.load(f)

        dict_config['env']['web_app_ctrl_fifo']['value'] = str(Path(env_dir, 'web_app_ctrl_fifo'))
        dict_config['env']['trainer_ctrl_fifo']['value'] = str(Path(env_dir, 'fifo_trainer_ctrl'))
        dict_config['env']['result_dir']['value'] = str(model.model_dir)
        dict_config['dataset']['dataset_dir']['value'] = str(Path(dataset_dir, f'dataset_{dataset.id}'))

        model.model_type = model.ai_model_sdk.model_type
        dict_config['model']['model_type']['value'] = model.model_type

        with open(Path(model.model_dir, 'config.json'), 'w') as f:
            json.dump(dict_config, f, ensure_ascii=False, indent=4)

        for fifo_path in [
            dict_config['env']['trainer_ctrl_fifo']['value'],
            dict_config['env']['web_app_ctrl_fifo']['value'],
        ]:
            if not Path(fifo_path).exists():
                os.mkfifo(fifo_path)

        self._set_step(job, 5, OperationStep.STATUS_DONE)

        # Step 6: Save dataset pickle into model dir
        self._set_step(job, 6, OperationStep.STATUS_RUNNING)
        model.dataset_pickle = str(Path(model.model_dir, 'dataset.pkl'))
        with open(model.dataset_pickle, 'wb') as f:
            pickle.dump(dataloader, f)

        model.status = model.STAT_IDLE
        model.save()
        self._set_step(job, 6, OperationStep.STATUS_DONE)

        # Step 7: Done
        self._set_step(job, 7, OperationStep.STATUS_DONE)
        self._finish(job, message='Model created')

    def _run_inference_run(self, job: OperationJob):
        model = job.model
        if model is None:
            raise RuntimeError('INFERENCE_RUN requires model')

        # Step 1: Validate
        self._set_step(job, 1, OperationStep.STATUS_RUNNING)
        config_path = Path(model.model_dir, 'config.json')
        if not config_path.exists():
            raise RuntimeError('config.json not found for model')
        sdk_path = Path(model.ai_model_sdk.ai_model_sdk_dir)
        if not sdk_path.exists():
            raise RuntimeError('AI Model SDK path not found')
        self._set_step(job, 1, OperationStep.STATUS_DONE)

        # Step 2: Prepare evaluation directory (clean previous outputs)
        self._set_step(job, 2, OperationStep.STATUS_RUNNING)
        evaluation_dir = Path(model.model_dir, 'evaluations')
        os.makedirs(evaluation_dir, exist_ok=True)
        for split in ('train', 'validation', 'test'):
            for suffix in ('prediction.json', 'prediction.csv', 'detection_summary.json', 'detection_predictions.json'):
                target = Path(evaluation_dir, f'{split}_' + suffix)
                if target.exists():
                    try:
                        target.unlink()
                    except Exception:
                        pass
            overlay_dir = Path(evaluation_dir, 'overlays', split)
            if overlay_dir.exists():
                for child in overlay_dir.glob('**/*'):
                    try:
                        if child.is_file():
                            child.unlink()
                    except Exception:
                        pass
        self._set_step(job, 2, OperationStep.STATUS_DONE)

        # Step 3: Run inference worker
        self._set_step(job, 3, OperationStep.STATUS_RUNNING)
        command = [
            sys.executable,
            str(Path(settings.BASE_DIR, 'app', 'machine_learning', 'ml_inference_main.py')),
            '--sdk_path', str(sdk_path),
            '--config', str(config_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, cwd=settings.BASE_DIR)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip()
            self._fail(job, RuntimeError('Inference worker failed'), order=3)
            # Record detail before exit
            OperationStep.objects.filter(job=job, order=3).update(detail=detail)
            return
        self._set_step(job, 3, OperationStep.STATUS_DONE)

        # Step 4: Verify outputs
        self._set_step(job, 4, OperationStep.STATUS_RUNNING)
        try:
            with open(config_path, 'r') as f:
                _cfg = json.load(f)
            task_value = _cfg.get('inference_parameter', {}).get('model', {}).get('task', {}).get('value', '')
        except Exception:
            task_value = ''

        is_detection = 'det' in str(task_value)
        missing = []
        for split in ('train', 'validation', 'test'):
            if is_detection:
                candidates = [
                    Path(evaluation_dir, f'{split}_detection_summary.json'),
                    Path(evaluation_dir, f'{split}_detection_predictions.json'),
                ]
            else:
                candidates = [Path(evaluation_dir, f'{split}_prediction.json')]

            if not any(p.exists() for p in candidates):
                missing.append('/'.join([p.name for p in candidates]))

        if missing:
            self._fail(job, RuntimeError(f'Missing prediction files: {", ".join(missing)}'), order=4)
            return
        self._set_step(job, 4, OperationStep.STATUS_DONE)

        # Step 5: Done
        self._set_step(job, 5, OperationStep.STATUS_DONE)
        self._finish(job, message='Inference completed')

    def handle(self, *args, **options):
        job_id = options['job_id']
        job = OperationJob.objects.prefetch_related('steps').select_related('project', 'model', 'dataset').get(id=job_id)

        if job.status in [OperationJob.STATUS_DONE, OperationJob.STATUS_ERROR, OperationJob.STATUS_CANCELED]:
            return

        job.status = OperationJob.STATUS_RUNNING
        job.save()

        try:
            if job.job_type == OperationJob.JOB_TYPE_PROJECT_CREATE:
                self._run_project_create(job)
            elif job.job_type == OperationJob.JOB_TYPE_MODEL_CREATE:
                self._run_model_create(job)
            elif job.job_type == OperationJob.JOB_TYPE_INFERENCE_RUN:
                self._run_inference_run(job)
            else:
                raise RuntimeError(f'Unsupported job_type: {job.job_type}')
        except Exception as exc:
            # Try to mark the first RUNNING step as ERROR
            running = job.steps.filter(status=OperationStep.STATUS_RUNNING).order_by('order').first()
            self._fail(job, exc, order=(running.order if running else None))
