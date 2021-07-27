import json
import time

from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.

def home(request):
    context = {
        'section':'main_section.html',
    };
    return render(request, 'index.html',context);

def dashboard1(request):
    context = {
        'section':'dashboard1.html',
    };
    return render(request, 'index.html',context);

def dashboard2(request):
    context = {
        'section':'dashboard2.html',
    };
    return render(request, 'index.html',context);

def dashboard3(request):
    context = {
        'section':'dashboard3.html',
    };
    return render(request, 'index.html',context);

def health01(request):
    context = {
        'section':'health01.html',
    };
    return render(request, 'index.html',context);

def health02(request):
    context = {
        'section':'health02.html',
    };
    return render(request, 'index.html',context);

def health03(request):
    context = {
        'section':'health03.html',
    };
    return render(request, 'index.html',context);

def health04(request):
    context = {
        'section':'health04.html',
    };
    return render(request, 'index.html',context);

def health05(request):
    context = {
        'section':'health05.html',
    };
    return render(request, 'index.html',context);

def health06(request):
    context = {
        'section':'health06.html',
    };
    return render(request, 'index.html',context);

def health07(request):
    context = {
        'section':'health07.html',
    };
    return render(request, 'index.html',context);

def health08(request):
    context = {
        'section':'health08.html',
    };
    return render(request, 'index.html',context);

def health09(request):
    context = {
        'section':'health09.html',
    };
    return render(request, 'index.html',context);

def health10(request):
    context = {
        'section':'health10.html',
    };
    return render(request, 'index.html',context);

def health11(request):
    context = {
        'section':'health11.html',
    };
    return render(request, 'index.html',context);