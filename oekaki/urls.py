from django.conf.urls import url
from . import views

app_name = 'oekaki'
urlpatterns = [
    url('', views.hello_template, name='hello_template'),
    url('image', views.predict, name='predict'),
]