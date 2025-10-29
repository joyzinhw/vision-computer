# modelo.py
import torch.nn as nn
from torchvision import models

class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = base.fc.in_features
        base.fc = nn.Linear(num_features, 1)
        self.model = base

    def forward(self, x):
        return self.model(x)
