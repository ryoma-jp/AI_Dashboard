import json
from pathlib import Path

from django import template
register = template.Library()

@register.filter
def in_project(models, project):
    """ Models filterd Project
    
    Filter to extract models are related ``project``
    """
    return models.filter(project=project)

@register.filter
def hist_lookup(value, args):
    """ Lookup for dict object of histogram
    
    Lookup for dict object of histogram.
    
    Args:
        value: object
        args (string): specify the strings comma separated
            - 1st block: feature name
            - 2nd block: axis ('hist_x' or 'hist_y')
    """
    
    if (args is not None):
        arg_list = [arg.strip() for arg in args.split(',')]
        feature_name = arg_list[0]
        axis = arg_list[1]
        
        return value[feature_name][axis]
    else:
        return None

@register.filter
def importance_lookup(value, arg):
    """ Lookup for dict object of feature importance
    
    Lookup for dict object of feature importance
    
    Args:
        value: object
        arg (string): specify the key
    """
    
    if (arg is not None):
        return value[arg]['importance']
    else:
        return None
    
@register.filter
def get_metrics(value):
    """ Get metrics
    
    Get metrics as dictionary object.
    ``key`` is metrics name and ``value`` is metrics value.
    
    Args:
        value: object of MlModel
    """
    
    metrics_file = Path(value.model_dir, 'evaluations', 'evaluations.json')
    
    if (metrics_file.exists()):
        with open(metrics_file, 'r') as f:
            dict_metrics = json.load(f)
    else:
        dict_metrics = {}
        
    return dict_metrics
