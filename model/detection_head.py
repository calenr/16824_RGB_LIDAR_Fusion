import torch
import torch.nn as nn
from .resnet import ResBlock

class YoloHead(nn.Module):
    def __init__(self, num_box_per_cell: int=1, box_length: int=9, input_dim: int=512):
        super(YoloHead, self).__init__()
        self.box_length = box_length
        self.output_dim = num_box_per_cell * box_length
        self.classifier = ResBlock(input_dim, self.output_dim)

    def forward(self, fused_feat: torch.Tensor):
        out = self.classifier(fused_feat)
        return out
