import os
import torch
import yaml
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
from .models import TrainingMetrics
from app.birdifyapp.tasks import run_training_task
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import uuid

# Import the training function (now a plain synchronous function)
from app.birdifyapp.tasks import run_training_task

from celery.result import AsyncResult

def task_status(request, task_id):
    result = AsyncResult(task_id)
    # result.info will contain your custom meta data if the task is in progress.
    return JsonResponse({
        'state': result.state,
        'info': result.info,
    })

def start_training_view(request):
    if request.method == 'POST':
        # Enqueue the training task asynchronously and capture the result.
        result = run_training_task.delay()
        # Return the task ID and a message as JSON.
        return JsonResponse({
            'task_id': result.id,
            'message': 'Training started. Check back later for the results.'
        })
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)
    
def classify_image(image_path):
    from models.efficientnet_v2_m import build_efficientnet_model
    model = build_efficientnet_model(num_classes=200, dropout_prob=0.3)
    checkpoint_path = os.path.join(settings.BASE_DIR, 'checkpoints', 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        return "No trained model found. Please run training first."
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return f"Error opening image: {e}"
    
    input_tensor = val_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    return f"Predicted class index: {pred.item()}"

def index(request):
    message = ""
    result_message = None
    form = ImageUploadForm(request.POST or None, request.FILES or None)
    
    if request.method == "POST":
        if "upload_image" in request.POST:
            # Your image upload & classification logic...
            pass
    
    metrics_list = TrainingMetrics.objects.all().order_by("-date")
    return render(request, "index.html", {
        "form": form,
        "message": message,
        "result": result_message,
        "metrics": metrics_list,
    })
    
# New view for classifying uploaded image
@csrf_exempt  # Alternatively, use CSRF token via JavaScript
def classify_image_view(request):
    if request.method == 'POST':
        # Process the uploaded image file.
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image uploaded.'}, status=400)
        
        uploaded_file = request.FILES['image']
        # Option 1: Save the file temporarily to disk
        temp_filename = os.path.join(settings.BASE_DIR, 'media', f"{uuid.uuid4().hex}_{uploaded_file.name}")
        os.makedirs(os.path.dirname(temp_filename), exist_ok=True)
        with open(temp_filename, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        # Get classification result.
        result = classify_image(temp_filename)
        
        # Optionally, remove the temporary file.
        os.remove(temp_filename)
        
        # Return the result as JSON.
        return JsonResponse({'result': result})
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)