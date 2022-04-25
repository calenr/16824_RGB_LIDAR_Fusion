import torch
import torch.nn as nn
from torchvision.transforms import Resize
from .image_encoder import ImgEncoder
from .point_pillars_net import PointCloudEncoder
from .detection_head import YoloHead
from .resnet import FusedFeatBackbone

class RgbLidarFusion(nn.Module):
    def __init__(self, args):
        super(RgbLidarFusion, self).__init__()
        self.args = args
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

        args.pc_grid_size = self.pc_enc.voxel_generator.grid_size
        self.image_resizer = Resize(args.pc_grid_size[1:3], antialias=True)

        self.classifier_input_size = args.pc_num_filters[-1] + 256

        self.fused_feat_cnn = FusedFeatBackbone(self.classifier_input_size)
        self.detection_head = YoloHead(args, args.pc_range, args.pc_voxel_size, args.pc_grid_size, 
                                       1, 9, self.classifier_input_size)


    def forward(self, image: torch.Tensor, lidar: list[torch.Tensor]) -> torch.Tensor:
        """
        :param image: (Batch, 3, 375, 1242)
        :param lidar: list of (N, 4), lenght of list must be equal to args.batch_size
        :return:
        """
        assert len(lidar) == self.args.batch_size

        # image feat is of shape (batch, 256, 24, 78)
        image_feat = self.img_enc(image)
        # point feat is of shape (batch, num_filters[-1], grid_size_y, grid_size_x)
        point_feat = self.pc_enc(lidar)
        # resized image feat is of shape (batch, 256, grid_size_y, grid_size_x)
        image_feat = self.image_resizer(image_feat)
        # fused feat is of shape (batch, num_filters[-1] + 256, grid_size_y, grid_size_x)
        fused_feat = torch.cat((point_feat, image_feat), dim=1)
        fused_feat = self.fused_feat_cnn(fused_feat)
        out = self.detection_head(fused_feat)

        # print(f"output shape: {out.shape}")

        return out
