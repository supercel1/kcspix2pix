from django.db import models

# Create your models here.

class Images(models.Model):
    image = models.ImageField(upload_to='pix2pix/')
    created_at = models.DateTimeField(auto_now=True)

class Files(models.Model):
    files = models.FileField(
        upload_to='cyclegan/',
        verbose_name='ファイル',
    )
    created_at = models.DateTimeField(auto_now=True)
