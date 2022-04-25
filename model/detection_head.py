import torch
import torch.nn as nn
from itertools import chain
from .resnet import ResBlock

CLASSNAMES_TO_IDX = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2
}

KITTILABEL_IDX = {
    "class": 0,
    "truncated": 1,
    "occluded": 2,
    "alpha": 3,
    "bbox_left": 4,
    "bbox_top": 5,
    "bbox_right": 6,
    "bbox_bottom": 7,
    "h": 8,
    "w": 9,
    "l": 10,
    "x": 11,
    "y": 12,
    "z": 13,
    "yaw": 14,
}

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
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, fused_feat: torch.Tensor):
        out = self.classifier(fused_feat)
        return out

    def calc_loss(self, output, target):
        return 0

    def convert_label_kitti_to_yolo(self, label: torch.Tensor):

        assert isinstance(label, torch.Tensor)
        assert len(label.shape) == 2
        assert label.shape[1] == 15

        yolo_labels = []

        for id in range(label.shape[0]):

            yolo_label = torch.zeros(self.box_length)
            kitti_label = label[id]

            if kitti_label[0] != CLASSNAMES_TO_IDX["Car"]:
                continue

            yolo_label[YOLOLABEL_IDX["conf"]] = 1
            yolo_label[YOLOLABEL_IDX["x"]] = kitti_label[KITTILABEL_IDX["x"]]
            yolo_label[YOLOLABEL_IDX["y"]] = kitti_label[KITTILABEL_IDX["y"]]
            yolo_label[YOLOLABEL_IDX["z"]] = kitti_label[KITTILABEL_IDX["z"]]
            yolo_label[YOLOLABEL_IDX["h"]] = kitti_label[KITTILABEL_IDX["h"]]
            yolo_label[YOLOLABEL_IDX["w"]] = kitti_label[KITTILABEL_IDX["w"]]
            yolo_label[YOLOLABEL_IDX["l"]] = kitti_label[KITTILABEL_IDX["l"]]
            yolo_label[YOLOLABEL_IDX["yaw_r"]] = torch.cos(kitti_label[KITTILABEL_IDX["yaw"]])
            yolo_label[YOLOLABEL_IDX["yaw_i"]] = torch.sin(kitti_label[KITTILABEL_IDX["yaw"]])

            yolo_labels.append(yolo_label)

        return torch.stack(yolo_labels)
