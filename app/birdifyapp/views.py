import os
import torch
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
from .models import TrainingMetrics
from app.birdifyapp.tasks import run_training_task
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import uuid
from models.efficientnet_v2_s import build_efficientnet_model
from torchvision import transforms


from celery.result import AsyncResult

def task_status(request, task_id):
    result = AsyncResult(task_id)
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

    # Build the model and load weights
    model = build_efficientnet_model(num_classes=200, dropout_prob=0.3)
    checkpoint_path = os.path.join(settings.BASE_DIR, 'checkpoints', 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        return "No trained model found. Please run training first."
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    # Define the validation transforms
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
    
    # Get the predicted index (0-indexed)
    predicted_index = pred.item()
    # Convert to 1-indexed to match the classes.txt numbering.
    class_id = predicted_index + 1

    # Define the path to your classes.txt file.
    classes_file = os.path.join("/content/data", "CUB_200_2011", "classes.txt")
    if not os.path.exists(classes_file):
        return f"Classes file not found at {classes_file}"

    # Parse classes.txt into a dictionary mapping the class number (1-indexed) to label.
    class_mapping = {}
    with open(classes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                num = int(parts[0])
            except ValueError:
                continue
            # Remove the numeric prefix and underscores
            label = " ".join(parts[1:]).replace("_", " ").lstrip("0123456789. ")
            class_mapping[num] = label

    predicted_label = class_mapping.get(class_id, f"Unknown class (ID {class_id})")

    # Return formatted output as "Index: Name"
    return f"Predicted class: {predicted_label} [{class_id}]"
    
@csrf_exempt 
def classify_image_view(request):
    if request.method == 'POST':
        # Process the uploaded image file.
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image uploaded.'}, status=400)
        
        uploaded_file = request.FILES['image']

        temp_filename = os.path.join(settings.BASE_DIR, 'media', f"{uuid.uuid4().hex}_{uploaded_file.name}")
        os.makedirs(os.path.dirname(temp_filename), exist_ok=True)
        with open(temp_filename, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        result = classify_image(temp_filename)
        
        os.remove(temp_filename)
        
        # Return the result as JSON.
        return JsonResponse({'result': result})
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=400)
    
def index(request):
    """
    Render the main page.
    """
    message = ""
    result_message = ""
    form = ImageUploadForm(request.POST or None, request.FILES or None)
    metrics_list = TrainingMetrics.objects.all().order_by("-date")
    return render(request, "index.html", {
        "form": form,
        "message": message,
        "result": result_message,
        "metrics": metrics_list,
    })