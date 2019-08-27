from django.conf.urls import url
from . import views

urlpatterns = [
    url('templates/', views.hello_template, name='hello_template'),
]