import torch
import torch.nn as nn
from .resnet import ResBlock

YOLOLABEL_IDX = {
    "conf": 0,
    "x": 1,
    "y": 2,
    "z": 3,
    "h": 4,
    "w": 5,
    "l": 6,
    "yaw_r": 7, # real part of yaw angle
    "yaw_i": 8, # imaginary part of yaw angle
}

class YoloHead(nn.Module):
    def __init__(self, num_box_per_cell: int=1, box_length: int=9, input_dim: int=512):
        super(YoloHead, self).__init__()
        self.box_length = box_length
        self.output_dim = num_box_per_cell * box_length
        self.classifier = ResBlock(input_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, fused_feat: torch.Tensor):
        out = self.classifier(fused_feat)
        out[:, YOLOLABEL_IDX["conf"], :, :] = torch.sigmoid(out[:, YOLOLABEL_IDX["conf"], :, :])
        # out[:, YOLOLABEL_IDX["h"]:YOLOLABEL_IDX["l"]+1, :, :] = torch.relu(out[:, YOLOLABEL_IDX["h"]:YOLOLABEL_IDX["l"]+1, :, :])
        return out
