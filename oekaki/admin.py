from django.contrib import admin

from .models import Images, Files, FakeImage
# Register your models here.
class ImageAdmin(admin.ModelAdmin):
    pass

class FilesAdmin(admin.ModelAdmin):
    pass

class FakeImageAdmin(admin.ModelAdmin):
    pass

admin.site.register(Images, ImageAdmin)
admin.site.register(Files, FilesAdmin)
admin.site.register(FakeImage, FakeImageAdmin)