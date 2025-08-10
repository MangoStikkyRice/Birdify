"""
This file contains a method mixup_data to blend two images and their labels to force
the model to learn from other features like body shape, beak size, etc. rather than
becoming overconfident on the training data.

Author: Jordan Miller

Sources:
    [1] https://medium.com/@lhungting/mixup-a-trivial-but-powerful-image-augmentation-technique-4e2d0725b8e3
"""

import torch
import numpy as np

def mixup_data(x, y, alpha=0.3):
    """
    Takes a batch of input data (x), true labels (y),
    and a hyperparameter (alpha) which controls the strength
    of the mixup. Note that x has shape (batch_size, C, H, W)
    """
    if alpha > 0:
        
        # This is beta distribution that samples a value between 1 and 0
        # skewed towards 0.5 since we use the same value for both params.
        lam = np.random.beta(alpha, alpha)
    else:
        
        # If alpha is zero or negative, then no mixup is applied.
        lam = 1
    batch_size = x.size()[0]
    
    # Every sample will be paired randomly with each other. Shuffle indices.
    index = torch.randperm(batch_size).to(x.device)

    # From [1], mix each sample with another random sample in the batch using lambda.
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # Combine labels
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
