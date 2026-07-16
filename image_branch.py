"""2D image branch for structural transform outputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageBranch(nn.Module):
    """2D branch for transformed spectral images."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.avg1, self.max1 = nn.AvgPool2d(2), nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 48, kernel_size=3, padding=1)
        self.avg2, self.max2 = nn.AvgPool2d(2), nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.avg3, self.max3 = nn.AvgPool2d(2), nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.avg4, self.max4 = nn.AvgPool2d(2), nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(128, 104, kernel_size=5, padding=2)
        self.fc1 = nn.LazyLinear(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = torch.cat((F.relu(self.avg1(x)), F.relu(self.max1(x))), 1)
        x = F.relu(self.conv2(x))
        x = torch.cat((F.relu(self.avg2(x)), F.relu(self.max2(x))), 1)
        x = F.relu(self.conv3(x))
        x = torch.cat((F.relu(self.avg3(x)), F.relu(self.max3(x))), 1)
        x = F.relu(self.conv4(x))
        x = torch.cat((F.relu(self.avg4(x)), F.relu(self.max4(x))), 1)
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))

