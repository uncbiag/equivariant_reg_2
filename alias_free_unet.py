import icon_registration.losses
import matplotlib.pyplot as plt
import time
import torch
import subprocess
import os
import footsteps
from icon_registration.networks import *


class NoDownsampleNet(nn.Module):
    def __init__(self, dimension=2, output_dim=128):
        super().__init__()
        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
        DIM = output_dim
        self.convs = nn.ModuleList([])
        self.batchnorms = nn.ModuleList([self.BatchNorm(DIM) for _ in range(3)])
        self.convs.append(self.Conv(1, DIM, 7, padding="same"))
        for i in range(6):
            self.convs.append(self.Conv(DIM, DIM, 7, padding="same"))

    def forward(self, x):
        x = self.convs[0](x)
        x = torch.relu(x)

        for i in range(3):
            x = self.batchnorms[i](x)
            y = self.convs[i + 1](x)
            y = torch.relu(y)
            y = self.convs[i + 4](y)

            x = y + x

        return x
