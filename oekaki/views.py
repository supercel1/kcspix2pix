from django.shortcuts import render

# Create your views here.

def hello_template(request):
    return render(request, 'index.html')

def predict(request):
    print(request)
    return render(request, 'predict.html')