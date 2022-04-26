import torch
import torch.nn as nn
from utils.kitti_viewer import Calibration

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

class YoloLoss(nn.Module):

    def __init__(self, args, pc_range: list[float], voxel_size: list[int]=[0.16, 0.16, 4],
                 num_box_per_cell: int=1, box_length: int=9):
        super(YoloLoss, self).__init__()
        self.box_length = box_length
        self.output_dim = num_box_per_cell * box_length
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.noobj_loss_weight = 0.5
        self.coor_loss_weight = 5
        self.voxel_size = voxel_size # XYZ
        self.x_offset = pc_range[0]
        self.y_offset = pc_range[1]
        self.device = args.device

    def forward(self, outputs: torch.Tensor, targets: list[torch.Tensor], calibs:list[Calibration], grid_size: list[int]):
        """
        grid size is in ZYX config
        """
        assert len(outputs.shape) == 4

        # aggregate tensors
        targets_flatten = []
        outputs_flatten = []
        obj_masks = []
        noobj_masks = []


        for batch_id in range(outputs.shape[0]):
            output = outputs[batch_id].to(self.device)
            target = targets[batch_id].to(self.device)
            calib = calibs[batch_id]
            output_grid_size = (output.shape[1], output.shape[2])  # YX
            # Ratio of the final embedding grid size vs. the original voxelized grid size
            grid_ratios = (output.shape[1] / grid_size[1], output.shape[2] / grid_size[2])  # YX

            # Convert the kitti target to yolo format, discarding non-car labels
            yolo_target = self.convert_label_kitti_to_yolo(target)

            # The target ground truth data casted to the appropriate grid location
            target_in_grid = torch.zeros_like(output).to(self.device)
            # Mask with the shape of the output grid
            obj_mask = torch.zeros(output_grid_size,dtype=torch.bool).to(self.device)

            if yolo_target is not None:
                yolo_target = yolo_target.to(self.device)
                # Get the xyz data of the target
                yolo_target_xyz_np = yolo_target[:, YOLOLABEL_IDX["x"]:YOLOLABEL_IDX["z"]+1].to('cpu').numpy()
                yolo_target_velo = yolo_target.clone()
                # Transform points to the velodyne frame
                yolo_target_velo[:, YOLOLABEL_IDX["x"]:YOLOLABEL_IDX["z"]+1] = torch.from_numpy(calib.project_rect_to_velo(yolo_target_xyz_np))

                # Holds the index of the grid that the target center falls in
                yolo_target_yx_idx = torch.zeros(yolo_target_velo.shape[0], 2, dtype=torch.long).to(self.device)

                # Index of y and x in the grid
                yolo_target_yx_idx[:, 0] = (yolo_target_velo[:, YOLOLABEL_IDX["y"]] - self.y_offset) / self.voxel_size[1] * grid_ratios[0]
                yolo_target_yx_idx[:, 1] = (yolo_target_velo[:, YOLOLABEL_IDX["x"]] - self.x_offset) / self.voxel_size[0] * grid_ratios[1]

                # Ensure not out of bound
                yolo_target_yx_idx[:, 0] = torch.clamp(yolo_target_yx_idx[:, 0], min=0, max=output_grid_size[0] - 1)
                yolo_target_yx_idx[:, 1] = torch.clamp(yolo_target_yx_idx[:, 1], min=0, max=output_grid_size[1] - 1)

                # Cells in the grid that has ground truth data is labelled true
                obj_mask[yolo_target_yx_idx] = True
                # Target info is copied to the one in grid format
                target_in_grid[:, yolo_target_yx_idx[:, 0], yolo_target_yx_idx[:, 1]] = torch.transpose(yolo_target, 0, 1)

            # no object mask for the noobj loss
            noobj_mask = obj_mask.logical_not()

            # Flatten and reshape for easy processing
            obj_mask = torch.reshape(torch.flatten(obj_mask), (-1, 1))
            noobj_mask = torch.reshape(torch.flatten(noobj_mask), (-1, 1))
            target_flatten = torch.reshape(target_in_grid, (-1, target_in_grid.shape[0]))
            output_flatten = torch.reshape(output, (-1, output.shape[0]))

            targets_flatten.append(target_flatten)
            outputs_flatten.append(output_flatten)
            obj_masks.append(obj_mask)
            noobj_masks.append(noobj_mask)
        
        targets_flatten = torch.cat(targets_flatten)
        outputs_flatten = torch.cat(outputs_flatten)
        obj_masks = torch.cat(obj_masks)
        noobj_masks = torch.cat(noobj_masks)

        # Calculate the losses
        obj_conf_loss = self.bce_loss(self.sigmoid(obj_masks * outputs_flatten[:, YOLOLABEL_IDX["conf"]]), obj_masks * targets_flatten[:, YOLOLABEL_IDX["conf"]])
        noobj_conf_loss = self.bce_loss(self.sigmoid(noobj_masks * outputs_flatten[:, YOLOLABEL_IDX["conf"]]), noobj_masks * targets_flatten[:, YOLOLABEL_IDX["conf"]])
        coord_loss = self.mse_loss(obj_masks * outputs_flatten[:, YOLOLABEL_IDX["x"]:YOLOLABEL_IDX["z"]+1], obj_masks * targets_flatten[:, YOLOLABEL_IDX["x"]:YOLOLABEL_IDX["z"]+1])
        shape_loss = self.mse_loss(obj_masks * outputs_flatten[:, YOLOLABEL_IDX["h"]:YOLOLABEL_IDX["l"]+1], obj_masks * targets_flatten[:, YOLOLABEL_IDX["h"]:YOLOLABEL_IDX["l"]+1])
        angle_loss = self.mse_loss(obj_masks * outputs_flatten[:, YOLOLABEL_IDX["yaw_r"]:YOLOLABEL_IDX["yaw_i"]+1], obj_masks * targets_flatten[:, YOLOLABEL_IDX["yaw_r"]:YOLOLABEL_IDX["yaw_i"]+1])

        return obj_conf_loss, self.noobj_loss_weight * noobj_conf_loss, self.coor_loss_weight * coord_loss, self.coor_loss_weight * shape_loss, self.coor_loss_weight * angle_loss

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

    def convert_label_yolo_to_kitti(self, label: torch.Tensor):
        assert isinstance(label, torch.Tensor)
        assert len(label.shape) == 2
        assert label.shape[1] == self.box_length

        kitti_labels = []

        for id in range(label.shape[0]):

            kitti_label = torch.zeros(15)
            yolo_label = label[id]

            kitti_label[KITTILABEL_IDX["class"]] = 0
            kitti_label[KITTILABEL_IDX["x"]] = yolo_label[YOLOLABEL_IDX["x"]]
            kitti_label[KITTILABEL_IDX["y"]] = yolo_label[YOLOLABEL_IDX["y"]]
            kitti_label[KITTILABEL_IDX["z"]] = yolo_label[YOLOLABEL_IDX["z"]]
            kitti_label[KITTILABEL_IDX["h"]] = yolo_label[YOLOLABEL_IDX["h"]]
            kitti_label[KITTILABEL_IDX["w"]] = yolo_label[YOLOLABEL_IDX["w"]]
            kitti_label[KITTILABEL_IDX["l"]] = yolo_label[YOLOLABEL_IDX["l"]]
            kitti_label[KITTILABEL_IDX["yaw"]] = torch.atan2(yolo_label[YOLOLABEL_IDX["yaw_i"]], yolo_label[YOLOLABEL_IDX["yaw_r"]])

            kitti_labels.append(kitti_label)

        if not kitti_labels:
            kitti_labels = None
        else:
            kitti_labels = torch.stack(kitti_labels)

        return kitti_labels
