from django.urls import path
from . import views

app_name = 'oekaki'
urlpatterns = [
    path('image', views.predict_image, name='predict_image'),
    path('file', views.predict_file, name='predict_file'),
    path('', views.index, name='index'),
]