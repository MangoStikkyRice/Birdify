"""
The seeds module contains a method set_seeds to ensure reproducibility of the system
by limiting the number of sources of nondeterministic behavior, this helps with
debugging and ensuring that the model behaves similarly across different runs.

Author: Jordan Miller

Source: https://pytorch.org/docs/stable/notes/randomness.html
"""

import random
import numpy as np
import torch


def set_seeds(seed):
    """Set seeds for all random generators to ensure reproducibility"""
    # Set seed for Python's random number generator, when we call Python's native
    # random functions like random.random(), it ensures those calls are reproducable.
    random.seed(seed)

    # Set seed for NumPy's random number generator.
    np.random.seed(seed)

    # Set seed for PyTorch's random number generator.
    torch.manual_seed(seed)

    # Set seeds for all GPUs being used by PyTorch.
    torch.cuda.manual_seed_all(seed)

    # Forces cuDNN to use deterministic algorithms when possible.
    torch.backends.cudnn.deterministic = True

    # Disables cuDNN auto-tuner, which finds the best algorithm for our hardware.
    torch.backends.cudnn.benchmark = False