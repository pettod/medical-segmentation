import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            *[nn.Conv2d(3, 1, 3, padding=1), nn.ReLU()])

    def forward(self, x):
        x = self.conv_1(x)
        return x
