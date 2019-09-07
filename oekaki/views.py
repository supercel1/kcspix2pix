from django.shortcuts import render
from django.http import JsonResponse, HttpResponse

import base64
import json

from .networks import CycleGAN
from .models import Images, Files

# Create your views here.

def index(request):
    print(request)
    return render(request, 'index.html')

def predict_image(request):
    model = CycleGAN()
    model.log_dir = 'logs'
    model.load('epoch195')

    print(request.POST)

    context = {'status': '200 OK'}
    return JsonResponse(context)

def predict_file(request):
    model = CycleGAN()
    model.log_dir = 'logs'
    model.load('epoch195')

    form = Files(request.POST, request.FILES)
    form.save()

    context = {'status': '200 OK'}
    return JsonResponse(context)