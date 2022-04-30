import os
import logging

from django.shortcuts import render, redirect

from views_common import SidebarActiveStatus, get_version

# Create your views here.

def view_streaming(request):
    """ Function: inference
     * inference top
    """
    sidebar_status = SidebarActiveStatus()
    sidebar_status.view_streaming = 'active'
    text = get_version()
        
    context = {
        'sidebar_status': sidebar_status,
        'text': text,
    }
    return render(request, 'view_streaming.html', context)

