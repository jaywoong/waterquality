"""config URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from dashboard import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('dashboard1', views.dashboard1, name='dashboard1'),
    path('dashboard2', views.dashboard2, name='dashboard2'),
    path('dashboard3', views.dashboard3, name='dashboard3'),
    path('health01', views.health01, name='health01'),
    path('health02', views.health02, name='health02'),
    path('health03', views.health03, name='health03'),
    path('health04', views.health04, name='health04'),
    path('health05', views.health05, name='health05'),
    path('health06', views.health06, name='health06'),
    path('health07', views.health07, name='health07'),
    path('health08', views.health08, name='health08'),
    path('health09', views.health09, name='health09'),
    path('health10', views.health10, name='health10'),
    path('health11', views.health11, name='health11'),

]
