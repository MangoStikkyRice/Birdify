import os
import sys
import yaml
from celery import shared_task

# Calculate the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Replicate initialization from main.py.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Stops oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Suppresses unnecessary TensorFlow logs

# Set seeds as in main.py.
from utils.seeds import set_seeds
set_seeds(42)

# Import your full training function.
# This function is defined in training/train.py and uses the Cub2011 dataset
# from your data module (data/data.py). The Cub2011 class downloads and extracts
# data from: https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1.
from training.train import train_model as real_train_model

@shared_task(bind=True)
def run_training_task(self):
    """
    Celery task that loads the configuration, replicates main.py's initialization,
    and runs the full training process (with progress updates) via train_model.
    """
    # Load configuration from configs/config.yaml.
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return f"Error loading config: {e}"
    
    # Determine total epochs from the configuration.
    total_epochs = config.get('training', {}).get('num_epochs', 20)
    
    # Define a progress callback to update the Celery task state.
    def progress_callback(current_epoch, metrics=None):
        """
        Called at the end of each epoch.
        :param current_epoch: The current epoch number (1-indexed).
        :param metrics: Dictionary of additional metrics (e.g., train_loss, val_loss, accuracy).
        """
        if metrics is None:
            metrics = {}
        self.update_state(
            state='PROGRESS',
            meta={
                'current_epoch': current_epoch,
                'total_epochs': total_epochs,
                **metrics
            }
        )
    
    # Call the real training function with the configuration and progress callback.
    # Note: The training loop (in training/train.py) uses the Cub2011 dataset
    # (defined in data/data.py) to load data from your configured root directory.
    result = real_train_model(config, progress_callback=progress_callback)
    
    return result
