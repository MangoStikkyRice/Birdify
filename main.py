"""
The entrypoint for training a model from ./models. This file contains one method that
sets the random number generator (RNG) seeds across the project

Author: Jordan Miller

Source: https://pytorch.org/docs/stable/notes/randomness.html
"""

import os

# Suppresses OneDNN optimization warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Control the verbosity of TensorFlow logging systems.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import yaml
from training.train import train_model
from utils.seeds import set_seeds

# From ./utils/seeds/set_seeds method, we set the seeds for RNGs.
set_seeds(42)

def main():
    """Calls training with the configuration file."""
    # Load configuration settings and hyperparameters from ./configs/config.yaml.
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Call training from ./training/train/train_model and pass in configurations.
    train_model(config)

if __name__ == '__main__':
    main()