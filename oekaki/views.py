from django.shortcuts import render
from django.http import JsonResponse, HttpResponse

import base64
import json

from . import models, networks

# Create your views here.

def hello_template(request):
    print(request)
    return render(request, 'index.html')

def predict(request):
    model = networks.CycleGAN()
    model.log_dir = '../logs'
    model.load('epoch195')

    img_base64 = request.POST.get('img')

    context = {'img': img_base64}
    return JsonResponse(context)