import logging
import shutil

from pathlib import Path

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings

from app.models import Project, AIModelSDK
from app.forms import AIModelSDKForm

from views_common import SidebarActiveStatus, get_version, get_jupyter_nb_url

# Create your views here.

def ai_model_sdk(request):
    """AI Model SDK
    """
    
    if (request.method == 'POST'):
        if ('ai_model_sdk_view_dropdown' in request.POST):
            logging.info('----------------------------------------')
            logging.info(f'[DEBUG] {request.method}')
            logging.info(f'[DEBUG] {request.POST}')
            logging.info(f'[DEBUG] {request.POST.getlist("ai_model_sdk_view_dropdown")}')
            logging.info('----------------------------------------')
            request.session['ai_model_sdk_view_dropdown_selected_project'] = request.POST.getlist('ai_model_sdk_view_dropdown')[0]
        
        elif ('ai_model_sdk_view_upload' in request.POST):
            form_ai_model_sdk = AIModelSDKForm(request.POST, request.FILES)
            if (form_ai_model_sdk.is_valid()):
                # --- get related project ---
                project_name = request.session.get('ai_model_sdk_view_dropdown_selected_project', None)
                project = Project.objects.get(name=project_name)
                
                # --- save model ---
                ai_model_sdk_model = AIModelSDK.objects.create(
                              name=form_ai_model_sdk.cleaned_data.get('name'),
                              project=project,
                              ai_model_sdk_zip=form_ai_model_sdk.cleaned_data.get('ai_model_sdk_zip'),
                          )
        
                # --- unzip ---
                logging.info('[INFO] unzip START')
                ai_model_sdk_dir = Path(ai_model_sdk_model.ai_model_sdk_zip.path).parent
                shutil.unpack_archive(ai_model_sdk_model.ai_model_sdk_zip.path, ai_model_sdk_dir)
                logging.info('[INFO] unzip DONE')

        return redirect('ai_model_sdk')
        
    else:
        project = Project.objects.all().order_by('-id').reverse()
        ai_model_sdk_objs = AIModelSDK.objects.all().order_by('-id').reverse()
        sidebar_status = SidebarActiveStatus()
        sidebar_status.ai_model_sdk = 'active'
        
        # check for existence of selected project name
        project_name_list = [p.name for p in project]
        selected_project_name = request.session.get('ai_model_sdk_view_dropdown_selected_project', None)
        
        if ((selected_project_name is not None) and (selected_project_name in project_name_list)):
            dropdown_selected_project = Project.objects.get(name=selected_project_name)
        else:
            dropdown_selected_project = None
        
        form_ai_model_sdk = AIModelSDKForm()
        
        logging.info('-------------------------------------')
        logging.info(project_name_list)
        logging.info(dropdown_selected_project)
        logging.info('-------------------------------------')
        context = {
            'project': project,
            'ai_model_sdk': ai_model_sdk_objs,
            'sidebar_status': sidebar_status,
            'text': get_version(),
            'jupyter_nb_url': get_jupyter_nb_url(),
            'dropdown_selected_project': dropdown_selected_project,
            'form_ai_model_sdk': form_ai_model_sdk,
        }
        return render(request, 'ai_model_sdk.html', context)

