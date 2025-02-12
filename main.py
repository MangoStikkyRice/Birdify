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
