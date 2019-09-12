from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.core.files.base import ContentFile

import uuid
import base64
import json
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

from .networks import CycleGAN
from .models import Images, File, FakeImage
from .forms import FileForm
from .preprocessing import make_input

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
    img_name = str(uuid.uuid4())
    image = ContentFile(img_data, img_name + '.png')

    Images.objects.create(image=image)
    image_path = 'media/pix2pix/' + img_name + '.png'

    img_input = make_input(image_path, img_name)
    fake_Y = model.G_X(img_input)

    fake_img_np = fake_Y[0].detach().numpy()
    fake_img_np = 0.5 * (fake_img_np + 1)
    fake_img_np = np.transpose(fake_img_np, (1, 2, 0))

    fake_img_pil = Image.fromarray((fake_img_np * 255).astype(np.uint8))

    fake_img_byte_array = io.BytesIO()
    fake_img_pil.save(fake_img_byte_array, format='JPEG')
    fake_img_data = fake_img_byte_array.getvalue()
    fake_img_name = 'fake_' + img_name + '.jpg'
    fake_image = ContentFile(fake_img_data, fake_img_name)

    FakeImage.objects.create(fake_image=fake_image)
    fake_image_path = 'media/pix2pix/fake/' + fake_img_name

    fake_ = Image.open(fake_image_path)

    context = {'fake_image_path': fake_image_path }
    return JsonResponse(context)

def predict_file(request):
    model = CycleGAN()
    model.log_dir = 'logs'
    model.load('epoch195')

    if request.method == 'POST':
        print(request.POST, request.FILES)
        form = FileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            context = {'form': form}
        else:
            context = {'message': '無効なフォームです'}
        return JsonResponse(context)
    else:
        form = FileForm()
        context = {'form': form}
        return JsonResponse(context)