import os
from celery import Celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.birdifysite.settings')

app = Celery('app.birdifysite')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
