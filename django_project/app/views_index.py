from django.shortcuts import render

from app.models import Project, Dataset, MlModel
from app.forms import ProjectForm, DatasetForm, MlModelForm

from views_common import SidebarActiveStatus, get_version, get_all_fifo_command, get_jupyter_nb_url

from machine_learning.lib.trainer.trainer import Trainer

# Create your views here.

def index(request):
    """ Function: index
     * show main view
    """
    get_all_fifo_command()
    
    projects = Project.objects.all().order_by('-id').reverse()
    project_form = ProjectForm()
    dataset = Dataset.objects.all().order_by('-id').reverse()
    dataset_form = DatasetForm()
    models = MlModel.objects.all().order_by('-id').reverse()
    
    sidebar_status = SidebarActiveStatus()
    sidebar_status.index = 'active'
    text = get_version()
    system_info = Trainer.GetSystemInfo()
    jupyter_nb_url = get_jupyter_nb_url()
    
    context = {
        'projects': projects,
        'project_form': project_form,
        'dataset': dataset,
        'dataset_form': dataset_form,
        'models': models,
        'sidebar_status': sidebar_status,
        'text': text,
        'jupyter_nb_url': jupyter_nb_url,
        'system_info': system_info,
    }
    return render(request, 'index.html', context)


