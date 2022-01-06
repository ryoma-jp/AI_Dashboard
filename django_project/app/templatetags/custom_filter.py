from django import template
register = template.Library()

""" Function: in_project
 * models filterd project
"""
@register.filter
def in_project(models, project):
    return models.filter(project=project)
