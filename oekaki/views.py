from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.core.files.base import ContentFile

import uuid
import base64
import json
import numpy as np

from .networks import CycleGAN
from .models import Images, Files

# Create your views here.

def index(request):
    return render(request, 'index.html')

def predict_image(request):
    model = CycleGAN()
    model.log_dir = 'logs'
    model.load('epoch195')

    img_byte = request.body
    img_base64 = img_byte.decode(encoding='utf-8')
    img_data = base64.b64decode(img_base64)
    img_name = str(uuid.uuid4()) + '.jpg'
    image = ContentFile(img_data, img_name)

    Images.objects.create(image=image)
    context = {'image_path': 'media/pix2pix/' + img_name}
    return JsonResponse(context)

def predict_file(request):
    model = CycleGAN()
    model.log_dir = 'logs'
    model.load('epoch195')

    form = Files(request.POST, request.FILES)
    form.save()

    context = {'status': '200 OK'}
    return JsonResponse(context)