import logging

from django.shortcuts import render, redirect

from app.models import Project, MlModel, Dataset

from views_common import SidebarActiveStatus, get_version

# Create your views here.

def inference(request):
    """ Function: inference
     * inference top
    """
    # logging.info('-------------------------------------')
    # logging.info(request.method)
    # logging.info(request.POST)
    # logging.info('-------------------------------------')
    if (request.method == 'POST'):
        if ('inference_view_project_dropdown' in request.POST):
            request.session['inference_view_selected_project'] = request.POST.getlist('inference_view_project_dropdown')[0]
                
        elif ('inference_view_model_dropdown' in request.POST):
            curr_project = Project.objects.get(name=request.session['inference_view_selected_project'])
            
            if 'inference_view_selected_model' in request.session.keys():
                prev_model = MlModel.objects.get(name=request.session['inference_view_selected_model'], project=curr_project)
            else:
                prev_model = None
            
            request.session['inference_view_selected_model'] = request.POST.getlist('inference_view_model_dropdown')[0]
            curr_model = MlModel.objects.get(name=request.session['inference_view_selected_model'], project=curr_project)
            
        elif ('inference_view_dataset_dropdown' in request.POST):
            curr_project = Project.objects.get(name=request.session['inference_view_selected_project'])
            
            request.session['inference_view_selected_dataset'] = request.POST.getlist('inference_view_dataset_dropdown')[0]
            curr_dataset = Dataset.objects.get(name=request.session['inference_view_selected_dataset'], project=curr_project)
            
        else:
            logging.warning('Unknown POST command:')
            logging.warning(request.POST)
        
        return redirect('inference')
    else:
        sidebar_status = SidebarActiveStatus()
        sidebar_status.inference = 'active'
        text = get_version()
        
        project = Project.objects.all().order_by('-id').reverse()
        project_name = request.session.get('inference_view_selected_project', None)
        if (project_name is not None):
            project_dropdown_selected = Project.objects.get(name=project_name)
        else:
            project_dropdown_selected = None
        
        if (project_dropdown_selected):
            # --- get model list and selected model ---
            model = MlModel.objects.filter(project=project_dropdown_selected).order_by('-id').reverse()
            
            model_name = request.session.get('inference_view_selected_model', None)
            if (model_name is not None):
                model_dropdown_selected = MlModel.objects.get(name=model_name, project=project_dropdown_selected)
            else:
                model_dropdown_selected = None
            
            # --- get dataset list and selected dataset ---
            dataset = Dataset.objects.filter(project=project_dropdown_selected).order_by('-id').reverse()
            
            dataset_name = request.session.get('inference_view_selected_dataset', None)
            if (dataset_name is not None):
                dataset_dropdown_selected = Dataset.objects.get(name=dataset_name, project=project_dropdown_selected)
            else:
                dataset_dropdown_selected = None
            
            
        else:
            model = MlModel.objects.all().order_by('-id').reverse()
            model_dropdown_selected = None
        
            dataset = Dataset.objects.all().order_by('-id').reverse()
            dataset_dropdown_selected = None
            
        context = {
            'project': project,
            'model': model,
            'dataset': dataset,
            'sidebar_status': sidebar_status,
            'text': text,
            'project_dropdown_selected': project_dropdown_selected,
            'model_dropdown_selected': model_dropdown_selected,
            'dataset_dropdown_selected': dataset_dropdown_selected,
        }
        return render(request, 'inference.html', context)

