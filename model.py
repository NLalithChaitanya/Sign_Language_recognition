import torch
import torch.nn as nn
import torch.nn.functional as F

    
class CNNModel3(nn.Module):
    def __init__(self, num_classes=24):
        super(CNNModel3, self).__init__()
        self.conv1 = nn.Conv2d(1, 80, kernel_size=5)    # -> 80x24x24
        self.bn1   = nn.BatchNorm2d(80)
        self.pool  = nn.MaxPool2d(2,2)                  # after pool -> 80x12x12

        self.conv2 = nn.Conv2d(80, 160, kernel_size=5)  # -> 160x8x8
        self.bn2   = nn.BatchNorm2d(160)
        # pool -> 160x4x4 = 2560

        self.fc1 = nn.Linear(160*4*4, 512)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)   # raw logits
        return x


