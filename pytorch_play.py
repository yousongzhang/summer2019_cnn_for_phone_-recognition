import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=48):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=0),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4*8*32, 48)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(48).to(device)
summary(model, (1, 25, 40))
print(model)

input = torch.randn(1, 1, 25, 40)
out = model(input)
print(out)
