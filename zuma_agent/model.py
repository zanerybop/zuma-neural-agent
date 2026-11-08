import torch
from torch import nn


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
            nn.Linear(384, 96), nn.ReLU(),
        )
        self.aim = nn.Sequential(nn.Linear(96, 2), nn.Sigmoid())
        self.action = nn.Linear(96, 3)

    def forward(self, frames):
        x = self.features(frames.float() / 255.0)
        return self.aim(x), self.action(x)
