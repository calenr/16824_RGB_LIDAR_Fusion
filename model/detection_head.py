import torch
import torch.nn as nn
from itertools import chain
from utils.kitti_viewer import Calibration
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

    def __init__(self, args, pc_range: list[float], voxel_size: list[int]=[0.16, 0.16, 4], grid_size: list[int]=[12, 12], 
                 num_box_per_cell: int=1, box_length: int=9, input_dim: int=512):
        super(YoloHead, self).__init__()
        self.box_length = box_length
        self.output_dim = num_box_per_cell * box_length
        self.classifier = ResBlock(input_dim, self.output_dim)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.noobj_loss_weight = 0.5
        self.coor_loss_weight = 5
        self.grid_size = grid_size # ZYX
        self.voxel_size = voxel_size # XYZ
        self.x_offset = pc_range[0]
        self.y_offset = pc_range[1]
        self.device = args.device

    def forward(self, fused_feat: torch.Tensor):
        out = self.classifier(fused_feat)
        return out

    def calc_loss(self, outputs: torch.Tensor, targets: list[torch.Tensor], calibs:list[Calibration]):
        assert len(outputs.shape) == 4
        assert outputs.shape[1] == self.output_dim

        # transform target from camera frame to velodyne frame
        batch_id = 0
        output = outputs[batch_id].to(self.device)
        target = targets[batch_id].to(self.device)
        calib = calibs[batch_id]
        output_grid_size = (output.shape[1], output.shape[2])  # YX
        grid_ratios = (output.shape[1] / self.grid_size[1], output.shape[2] / self.grid_size[2])  # YX

        yolo_target = self.convert_label_kitti_to_yolo(target)

        target_in_grid = torch.zeros_like(output).to(self.device)
        obj_mask = torch.zeros(output_grid_size,dtype=torch.bool).to(self.device)
        
        
        if yolo_target is not None:
            yolo_target = yolo_target.to(self.device)
            yolo_target_xyz_np = yolo_target[:, YOLOLABEL_IDX["x"]:YOLOLABEL_IDX["z"]+1].to('cpu').numpy()
            yolo_target_velo = yolo_target.clone()
            yolo_target_velo[:, YOLOLABEL_IDX["x"]:YOLOLABEL_IDX["z"]+1] = torch.from_numpy(calib.project_rect_to_velo(yolo_target_xyz_np))

            yolo_target_yx_idx = torch.zeros(yolo_target_velo.shape[0], 2, dtype=torch.long).to(self.device)

            # index of y and x in the grid
            yolo_target_yx_idx[:, 0] = (yolo_target_velo[:, YOLOLABEL_IDX["y"]] - self.y_offset) / self.voxel_size[1] * grid_ratios[0]
            yolo_target_yx_idx[:, 1] = (yolo_target_velo[:, YOLOLABEL_IDX["x"]] - self.x_offset) / self.voxel_size[0] * grid_ratios[1]

            obj_mask[yolo_target_yx_idx] = True

            target_in_grid[:, yolo_target_yx_idx[:, 0], yolo_target_yx_idx[:, 1]] = torch.transpose(yolo_target, 0, 1)

        noobj_mask = obj_mask.logical_not()
        obj_mask = torch.flatten(obj_mask)
        noobj_mask = torch.flatten(noobj_mask)

        target_flatten = torch.reshape(target_in_grid, (target_in_grid.shape[0], -1))
        output_flatten = torch.reshape(output, (output.shape[0], -1))
        obj_conf_loss = self.bce_loss(obj_mask * output_flatten[YOLOLABEL_IDX["conf"], :], obj_mask * target_flatten[YOLOLABEL_IDX["conf"], :])
        noobj_conf_loss = self.bce_loss(noobj_mask * output_flatten[YOLOLABEL_IDX["conf"], :], noobj_mask * target_flatten[YOLOLABEL_IDX["conf"], :])
        coord_loss = self.mse_loss(obj_mask * output_flatten[YOLOLABEL_IDX["x"]:YOLOLABEL_IDX["z"]+1, :], obj_mask * target_flatten[YOLOLABEL_IDX["x"]:YOLOLABEL_IDX["z"]+1, :])
        shape_loss = self.mse_loss(torch.sqrt(obj_mask * output_flatten[YOLOLABEL_IDX["h"]:YOLOLABEL_IDX["l"]+1, :]), torch.sqrt(obj_mask * target_flatten[YOLOLABEL_IDX["h"]:YOLOLABEL_IDX["l"]+1, :]))

        return self.coor_loss_weight * (coord_loss + shape_loss) + obj_conf_loss + self.noobj_loss_weight * noobj_conf_loss

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

        if not yolo_labels:
            yolo_labels = None
        else:
            yolo_labels = torch.stack(yolo_labels)

        return yolo_labels
