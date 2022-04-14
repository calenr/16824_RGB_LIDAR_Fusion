import torch
import torch.nn as nn


class RgbLidarFusion(nn.Module):
    def __init__(self, args):
        super(RgbLidarFusion, self).__init__()

        self.classifier = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return x
