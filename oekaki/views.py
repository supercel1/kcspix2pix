from django.shortcuts import render

import base64

from . import models, networks

# Create your views here.

def hello_template(request):
    return render(request, 'index.html')

def predict(request):
    # model = networks.CycleGAN()
    # model.log_dir = '../logs'
    # model.load('epoch195')

    img_base64 = request['img']

    context = {'img': img_base64}
    return render(request, 'predict.html', context)