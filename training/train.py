import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import the dataset, model builder, and augmentation helper
from data.datasets import Cub2011
from models.efficientnet_v2_m import build_efficientnet_model
from utils.augmentation import mixup_data

# Import the metrics helper
from utils.metrics import compute_metrics

def train_model(config):
    # Setup device, seeds, etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transforms (could also be moved to a separate module)
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dataset = Cub2011(root=config['data']['root'], train=True, transform=train_transform, download=True)
    val_dataset   = Cub2011(root=config['data']['root'], train=False, transform=val_transform, download=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, 
                              num_workers=config['data']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, 
                            num_workers=config['data']['num_workers'], pin_memory=True)
    
    # Build model
    model = build_efficientnet_model(num_classes=config['model']['num_classes'], dropout_prob=config['model']['dropout'])
    model = model.to(device)
    
    # Training components
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['training']['max_lr'], steps_per_epoch=len(train_loader), epochs=config['training']['num_epochs']
    )
    scaler = torch.amp.GradScaler()
    writer = SummaryWriter(log_dir=config['training']['log_dir'])
    
    best_val_accuracy = 0.0
    
    save_dir = config['training']['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Training loop
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.4)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_dataset)
        print(f"Training Loss: {epoch_loss:.4f}")
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = val_loss / len(val_dataset)
        accuracy, f1, precision, recall, report = compute_metrics(all_labels, all_preds)
        print(f"Validation Loss: {epoch_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        writer.add_scalar("Loss/Validation", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)
        
        print("Classification Report:")
        print(report)
        
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("Best model updated.")
    
    print("\nTraining complete!")
    writer.close()
