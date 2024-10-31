import torch
import torch.nn as nn
from torchvision import models

class UAVClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(UAVClassifier, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Use pretrained ResNet18
        # Replace the final fully connected layer with one for our number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
