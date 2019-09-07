from django.urls import path
from . import views

app_name = 'oekaki'
urlpatterns = [
    path('image', views.predict, name='predict'),
    path('', views.hello_template, name='hello_template'),
]