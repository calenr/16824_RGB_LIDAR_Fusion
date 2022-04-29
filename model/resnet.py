import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        super(ResBlock, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 2, bias=False),
            nn.BatchNorm2d(output_channel),
        )

        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
            nn.Conv2d(output_channel, output_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, 512, grid_size_y, grid_size_x)
        :return: 
        """
        return self.layers(x) + self.downsample(x)


class FusedFeatBackbone(nn.Module):
    def __init__(self, input_channel: int):
        super(FusedFeatBackbone, self).__init__()

        self.backbone = nn.Sequential(
            ResBlock(input_channel, input_channel),
            ResBlock(input_channel, input_channel),
            ResBlock(input_channel, input_channel),
            ResBlock(input_channel, input_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, 512, grid_size_y, grid_size_x)
        :return: 
        """
        x = self.backbone(x)
        return x
