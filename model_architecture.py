import torch
import torch.nn as nn

class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=stride, padding=2, groups=in_channels)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        x = self.pool(x)
        return self.activation(x)

class PouleDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2)
        self.block1 = BlazeBlock(24, 24)
        self.block2 = BlazeBlock(24, 48)
        self.block3 = BlazeBlock(48, 48)
        self.block4 = BlazeBlock(48, 96)
        self.block5 = BlazeBlock(96, 96)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.sigmoid(x)

