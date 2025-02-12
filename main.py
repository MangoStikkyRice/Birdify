import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Stops oneDNN warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses unnecessary TensorFlow logs
import yaml
from training.train import train_model
from utils.seeds import set_seeds

set_seeds(42)

def main():
    # Load configuration from YAML file
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Kick off the training process
    train_model(config)

if __name__ == '__main__':
    main()
