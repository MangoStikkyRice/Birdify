import os
import sys
import yaml
from celery import shared_task
from utils.seeds import set_seeds
from training.train import train_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

set_seeds(42)

@shared_task(bind=True)
def run_training_task(self):
    """
    Celery task that loads the configuration and runs the training process
    """
    # Load configuration from configs/config
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return f"Error loading config: {e}"
    
    total_epochs = config.get('training', {}).get('num_epochs', 20)
    
    # Define a progress callback to update the task state.
    def progress_callback(current_epoch, metrics=None):
        """
        Called at the end of each epoch.
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

    result = train_model(config, progress_callback=progress_callback)
    
    return result
