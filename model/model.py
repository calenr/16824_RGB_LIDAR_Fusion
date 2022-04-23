import torch
import torch.nn as nn
from .image_encoder import ImgEncoder
from .point_pillars_net import PointCloudEncoder


class RgbLidarFusion(nn.Module):
    def __init__(self, args):
        super(RgbLidarFusion, self).__init__()

        self.img_enc = ImgEncoder()
        self.pc_enc = PointCloudEncoder(
            num_input_features=args.pc_num_input_features,
            use_norm=args.pc_use_norm,
            num_filters=args.pc_num_filters,
            with_distance=args.pc_with_distance,
            voxel_size=args.pc_voxel_size,
            pc_range=args.pc_range,
            max_num_voxels=args.pc_max_num_voxels,
            max_num_points_per_voxel=args.pc_max_num_points_per_voxel,
            batch_size=args.batch_size,
            device=args.device,
        )

    def forward(self, image: torch.Tensor, lidar: list[torch.Tensor]) -> torch.Tensor:
        """
        :param image: Batch x 3 x
        :param lidar: Batch x N x 3
        :return:
        """
        image_feat = self.img_enc(image)
        point_feat = self.pc_enc(lidar)

        print(image_feat.shape)
        print(point_feat.shape)

        return image_feat
