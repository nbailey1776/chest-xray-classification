import torch.nn as nn
import torchvision.models as models


def create_model():
    model = models.densenet121(pretrained=True)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 15),
        nn.Sigmoid()
    )
    return model

