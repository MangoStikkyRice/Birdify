import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

def build_efficientnet_model(num_classes=200, dropout_prob=0.3):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    num_features = model.classifier[-1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_prob),
        nn.Linear(num_features, num_classes)
    )
    return model
