import os
from celery import Celery

# Set the default Django settings module.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.birdifysite.settings')

app = Celery('app.birdifysite')

# Load any custom configuration from Django settings, using a CELERY_ namespace.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Autodiscover tasks from installed apps.
app.autodiscover_tasks()
