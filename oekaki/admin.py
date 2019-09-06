from django.contrib import admin

from .models import Images
# Register your models here.
class ImageAdmin(admin.ModelAdmin):
    pass

admin.site.register(Images, ImageAdmin)