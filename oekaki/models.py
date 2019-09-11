from django.db import models

# Create your models here.

class Images(models.Model):
    image = models.ImageField(upload_to='pix2pix/')
    created_at = models.DateTimeField(auto_now=True)

class File(models.Model):
    files = models.ImageField(upload_to='cyclegan/')
    created_at = models.DateTimeField(auto_now=True)

class FakeImage(models.Model):
    fake_image = models.ImageField(upload_to='pix2pix/fake/')
    created_at = models.DateTimeField(auto_now=True)
