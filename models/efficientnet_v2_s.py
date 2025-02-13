"""
Builds the EfficientNet V2 (Small) model

Author: Jordan Miller

Sources:
    [1] https://medium.com/@ivansanchez911/dropout-a-brief-review-4183d50a764a
"""

import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# The research from [1] talks about how dropout probabilities greater than 0.5 are
# quite aggressive for CNNs, which is expected, but we'll just start with 0.3 purely
# as a guestimation.
def build_efficientnet_model(num_classes=200, dropout_prob=0.3):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    num_features = model.classifier[-1].in_features

    # As part of transfer learning, we replace the classification layer, which
    # originally had 1000 classes. We set a dropout layer to disable a percentage
    # of neurons randomly during training. We also set a fully connected layer that
    # maps num_features to num_classes, which is 200 bird species.
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_prob),
        nn.Linear(num_features, num_classes)
    )
    return model
