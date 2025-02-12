"""
URL configuration for birdifysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from app.birdifyapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('app.birdifyapp.urls')),
    path('classify-image/', views.classify_image_view, name='classify-image'),
    # Add a URL for starting training via AJAX.
    path('start-training/', views.start_training_view, name='start-training'),
    # You’ll also need a URL to poll task status (see next step).
    path('task-status/<str:task_id>/', views.task_status, name='task-status'),
]
