"""
The entrypoint for training a model from ./models. This file contains one method that
sets the random number generator (RNG) seeds across the project

Author: Jordan Miller

Sources:
    [1] https://discuss.pytorch.org/t/dimensions-of-an-input-image/19439
    [2] https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    [3] https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    [4] https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
    [5] https://pytorch.org/docs/stable/tensorboard.html
    [6] https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    [7] https://tqdm.github.io/
"""
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import transforms


from data.datasets import Cub2011
from models.efficientnet_v2_s import build_efficientnet_model
from utils.augmentation import mixup_data
from utils.metrics import compute_metrics
# from utils.visualization import visualize_predictions

def train_model(config, progress_callback=None):
    """
    Trains the model using the configuration from ../configs/config.
    Optionally calls progress_callback(current_epoch, metrics) at the end of each epoch.
    """
    
    # Checks if CUDA is available for tensors and models to run on the GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms:
    #   1. Random resized crop forces the model to learn different parts of the bird.
    #   2. Color jitter helps improve generalization of birds in different lighting.
    # Then convert images to PyTorch tenors, which have the format (C, H, W) as per [1].
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
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

    # We defined a dataset class in ../data/datasets/Cub2011, which is a class
    # with methods to download and check the integrity of the files.
    # Here, we just construct the train and validation datasets for training.
    train_dataset = Cub2011(root=config['data']['root'], train=True, transform=train_transform, download=True)
    val_dataset   = Cub2011(root=config['data']['root'], train=False, transform=val_transform, download=False)
    
    # From [2] we use DataLoaders. While training a model, we typically want to
    # pass samples as minibatches, reshuffle the data, and maybe use multiprocessing.
    # These DataLoaders are just iterables that abstract this complexity for us.
    train_loader = DataLoader(train_dataset,
                              batch_size=config['training']['batch_size'],
                              shuffle=True,
                              num_workers=config['data']['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config['training']['batch_size'],
                            shuffle=False,
                            num_workers=config['data']['num_workers'],
                            pin_memory=True)
    
    # We use the method from ../models/efficientnet_v2_X/build_efficientnet_model
    # to build the neural network, which is just a bunch of layers with neurons
    # which have weights and biases.
    model = build_efficientnet_model(num_classes=config['model']['num_classes'],
                                     dropout_prob=config['model']['dropout'])
    
    # Move the model to either the CPU or GPU depending on availability.
    model = model.to(device)
    
    # This is a standard loss function for multi-class classification. We have some
    # label smoothing so that instead of applying 100% probability to the correct class,
    # we spread 10% among other classes.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # The optimizer is what adjusts weights to reduce loss. It updates parameters
    # based on gradients calculated during backpropagation [3].
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['training']['lr'],
                            weight_decay=config['training']['weight_decay'])
    
    # This is a learning rate scheduler that adjusts the learning rate over training.
    # Bad learning rates cause the model to either train too slowly or miss the
    # optimal solution. One-cycle learning rate policies increase learning rate at
    # the start and then gradually decrease; this leads to faster convergence and
    # avoids local minima.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['max_lr'],
        steps_per_epoch=len(train_loader),
        epochs=config['training']['num_epochs']
    )
    
    # Deep learning models typically use 32-bit floating point precision. Automatic
    # Mixed Precision with GradScaler speeds up training using 16-point where possible.
    # This is actually quite crucial as the model may fail to converge without it.
    # [4] explains this succinctly.
    # But anyway, GradScaler is a scaling utility that modifies gradients before
    # backpropagation to prevent underflow.
    scaler = torch.amp.GradScaler()
    
    # Log training data to TensorBoard. These are all saved to ../runs.
    # More details at [5].
    writer = SummaryWriter(log_dir=config['training']['log_dir'])

    # Store the highest accuracy during training; saves the best model only.
    best_val_accuracy = 0.0
    save_dir = config['training']['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    total_epochs = config['training']['num_epochs']
    
    # Training loop: Iterate over the number of epochs specified in ../configs/config
    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        
        # Set model to training mode using the train() method from Pytorch.
        # This activates layers like dropout and batch normalization [6].
        model.train()
        running_loss = 0.0
        
        # Just a loading bar. More details from [7].
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        
        # Iterate over each batch provided by train_loader.
        for inputs, labels in train_bar:
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixup augmentation from ../utils/augmentation/mixup_data.
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.4)
            
            optimizer.zero_grad()
            
            # Enable AMP and pass the input batch to the model to make predictions.
            with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                outputs = model(inputs)
                
                # This calculates loss, but we need to consider our mixup augmentation.
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
        
        # This is for TensorBoard by the way.
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        
        # Switch the model to eval mode. It's the same as the training mode, except
        # the dropout layers are disabled and the batch normalization layers use
        # the running statistics instead of the batch's.
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        
        # You do not need gradients for evaluation. Saves some memory.
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # Take index of highest scoring class. That's the prediction.
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = val_loss / len(val_dataset)
        
        # Use the method from ../utils/metrics/compute_metrics
        accuracy, _, _, _, report = compute_metrics(all_labels, all_preds)
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
        
        # Call the progress callback after each epoch if provided and update it.
        if progress_callback is not None:
            progress_callback(epoch + 1, {
                'train_loss': epoch_loss,
                'val_loss': epoch_val_loss,
                'accuracy': accuracy,
            })
    
    # Just make sure to close this since it blocks the script.
    # visualize_predictions(model, val_dataset, device, n_samples=6)
    print("\nTraining complete!")
    writer.close()
    return "Training completed successfully."

if __name__ == "__main__":
    # Only run the training loop if this file is executed directly.
    import yaml
    config_path = "path/to/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    train_model(config)