from django.contrib import admin

from .models import Images, File, FakeImage
# Register your models here.
class ImageAdmin(admin.ModelAdmin):
    pass

class FileAdmin(admin.ModelAdmin):
    pass

class FakeImageAdmin(admin.ModelAdmin):
    pass

admin.site.register(Images, ImageAdmin)
admin.site.register(File, FileAdmin)
admin.site.register(FakeImage, FakeImageAdmin)