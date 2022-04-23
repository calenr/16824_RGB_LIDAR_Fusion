from multiprocessing.context import assert_spawning
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch.utils import PointToVoxel


def get_voxel_gen(vsize_xyz: list[float] = [0.16, 0.16, 4.0],
                  coors_range_xyz: list[float] = [0, -40, -3, 70.4, 40, 1], num_point_features: int = 4,
                  max_num_voxels: int = 12000, max_num_points_per_voxel: int = 100,
                  device: torch.device = torch.device("cuda")
                  ):
    """
    returns voxel generator, use it by calling the forward function
    :input pointcloud: unstructed pointcloud of shape (N, num_point_features), ordering is XYZR
    :return voxels: voxelized pointcloud data. shape: (num_voxel, max_num_points_per_voxel, num_point_features)
    :return indices: indices of which voxel a point belongs to. shape: (num_voxel, 3), ordering is ZYX
    :return num_points: number of points that are inside each voxel. shape: (num_voxel)

    :param vsize_xyz: voxel size in x,y,z e.g. [0.16, 0.16, 4.0]
    :param coors_range_xyz: point cloud range values [xmin, ymin, zmin, xmax, ymax, zmax]
    :param num_point_features: dimension of the pointcloud data, 4 if xyzr, 3 if xyz
    :param max_num_voxels:
    :param max_num_points_per_voxel:
    :param device:
    """
    return PointToVoxel(
        vsize_xyz=vsize_xyz, coors_range_xyz=coors_range_xyz,
        num_point_features=num_point_features, max_num_voxels=max_num_voxels,
        max_num_points_per_voxel=max_num_points_per_voxel, device=device
    )


class Passthrough(nn.Module):
    """
    Does nothing, just passthrough the input
    """

    def __init__(self):
        super(Passthrough, self).__init__()

    def forward(self, x):
        return x


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False
                 ):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super(PFNLayer, self).__init__()
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.relu = nn.ReLU()

        if use_norm:
            self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=1e-2)
            self.linear = nn.Linear(in_channels, self.units, bias=False)
        else:
            self.norm = Passthrough()
            self.linear = nn.Linear(in_channels, self.units, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = self.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.16, 0.16, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)
                 ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super(PillarFeatureNet, self).__init__()
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            # Create networks of I/O dim, e.g.
            # (9, 64)
            # (64, 128)
            # (128, 256)
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features: torch.Tensor, indices: torch.Tensor, num_points_per_voxel: torch.Tensor) -> torch.Tensor:
        """
        :param features: voxelized pointcloud (num_voxel, max_num_points_per_voxel, 4) (x, y, z, r)
        :param indices: (num_voxels, 3), the index of pillar that the points belong to, ZYX ordering
        :param num_points_per_voxel: (num_voxels), the actual number of points in each voxel, because of 0 padding
        :return: embedding tensor: (num_voxel, num_filters[-1])
        """
        assert len(features.shape) == 3
        assert len(indices.shape) == 2
        assert indices.shape[1] == 3
        assert len(num_points_per_voxel.shape) == 1

        dtype = features.dtype

        # Find the average xyz per voxel
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_points_per_voxel.type_as(features).unsqueeze(1).unsqueeze(1)
        # Calculate Xc, Yc, Zc, offset from arithmetic mean
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        # Calculate Xp, x offset from pillar center
        f_center[:, :, 0] = features[:, :, 0] - (
            indices[:, 2].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        # coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        # Calculate Yp, y offset from pillar center
        f_center[:, :, 1] = features[:, :, 1] - (
            indices[:, 1].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
        # coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # After augmentation, the shape is now
        # [num_voxel, max_num_points_per_voxel, 9] (x, y, z, r, xc, yc, zc, xp, yp)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(
            num_points_per_voxel, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        # output shape is [num_voxel, embedding feature size]
        # embedding size is the last number of num_filters
        return features.squeeze()


def get_paddings_indicator(actual_num: torch.Tensor, max_num: int, axis=0):
    """
    Create boolean mask to zero the padding points.
    :param actual_num: the actual number of points in each voxel
    :param max_num: int, the max_num_points_per_voxel
    :return: boolean mask: ()
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class PointPillarsPseudoImage(nn.Module):
    def __init__(self, output_shape=(16, 64, 200, 200)):
        """
        Converts learned features from dense tensor to sparse pseudo image.
        :param output_shape: ([int]: 4). (batch_size, num_input_features, height, width)
        :param num_input_features: <int>. Number of input embedding features.
        """

        super(PointPillarsPseudoImage, self).__init__()
        self.name = 'PointPillarsPseudoImage'
        self.output_shape = output_shape

        self.batch_size = output_shape[0]
        self.nchannels = output_shape[1]
        # Coordinates are in top-down view, y is the lateral dimension of the ego (left-right)
        # x is the longitudinal dimension of the ego (front-back)
        self.ny = output_shape[2]
        self.nx = output_shape[3]

    def forward(self, voxel_features: torch.Tensor, batched_indices: torch.Tensor) -> torch.Tensor:
        """
        :param voxel_features: batched pc embedding (SUM(num_voxel), num_input_features). We concatenate the batches, not stack along new dimension
        :param batched_indices: (num_voxels, 4). The index of pillar that the points belong to. The first element is batch number, the rest are ZYX
        :return: embedding tensor pseudo-image: (batch_size, num_input_features, height, width)
        """

        assert len(voxel_features.shape) == 2
        assert batched_indices.shape[1] == 4

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(self.batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros((self.nchannels, self.nx * self.ny),
                                 dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = batched_indices[:, 0] == batch_itt
            this_batched_indices = batched_indices[batch_mask, :]
            indices = this_batched_indices[:, 2] * \
                self.nx + this_batched_indices[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(self.batch_size, self.nchannels, self.ny,
                                         self.nx)
        return batch_canvas


class PointCloudEncoder(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.16, 0.16, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 max_num_voxels=12000,
                 max_num_points_per_voxel=100,
                 batch_size=2,
                 device=torch.device("cuda")
                 ):
        super(PointCloudEncoder, self).__init__()

        # Converts single frame pointcloud (L, 4) to
        self.voxel_generator = get_voxel_gen(
            vsize_xyz=voxel_size,
            coors_range_xyz=pc_range,
            num_point_features=num_input_features,
            max_num_voxels=max_num_voxels,
            max_num_points_per_voxel=max_num_points_per_voxel,
            device=device
        )

        # Converts single frame voxels (P, N, 3) to 2D learned feat (P, C)
        self.pillar_feat_net = PillarFeatureNet(
            num_input_features=num_input_features,
            use_norm=use_norm,
            num_filters=num_filters,
            with_distance=with_distance,
            voxel_size=voxel_size,
            pc_range=pc_range
        ).to(device)

        self.grid_size = self.voxel_generator.grid_size
        self.batch_size = batch_size

        # Converts batch 2D learned feat (B, P, C) to batch pseudo-images (B, C, H, W)
        self.pseudoimage_net = PointPillarsPseudoImage(
            output_shape=(
                self.batch_size, num_filters[-1], self.grid_size[1], self.grid_size[2])
        ).to(device)

        self.device = device

    def forward(self, pointcloud_list: list[torch.Tensor]) -> torch.Tensor:
        """
        :param pointcloud_list: list of (num_points x 4) for xyzr of length batch_size
        :return pseudo-images: (batch_size, num_filters[-1], grid_size_y, grid_size_x)
        """

        assert len(pointcloud_list) == self.batch_size

        learned_feat_list = []
        indices_batch_list = []
        for batch_id, pointcloud in enumerate(pointcloud_list):
            voxels, indices, num_points_per_voxel = self.voxel_generator(
                pointcloud.to(self.device))
            learned_feat = self.pillar_feat_net(
                voxels, indices, num_points_per_voxel)
            learned_feat_list.append(learned_feat)

            batch_info = torch.full(
                [indices.shape[0], 1], fill_value=batch_id).to(self.device)
            indices_w_batch_info = torch.cat((batch_info, indices), dim=1)
            indices_batch_list.append(indices_w_batch_info)

        learned_feats = torch.cat(learned_feat_list, dim=0)
        indices_batches = torch.cat(indices_batch_list, dim=0)

        pseudo_images = self.pseudoimage_net(learned_feats, indices_batches)

        return pseudo_images


if __name__ == '__main__':
    device = torch.device("cuda")

    ###
    # print("Sparse lidar example")
    # voxel_size = [0.1, 0.1, 5.0]
    # coors_range = [-10.0, -10.0, -10.0, 10.0, 10.0, 10.0]
    # num_point_features = 4
    # max_num_points_per_voxel = 5

    # voxel_generator = get_voxel_gen(
    #     vsize_xyz=voxel_size, coors_range_xyz=coors_range, num_point_features=num_point_features,
    #     max_num_voxels=20000, max_num_points_per_voxel=max_num_points_per_voxel, device=device
    # )
    # # dummy pointcloud xyz between -10 and 10
    # lidar_xyz = torch.rand(size=[1000, 3]) * 20 - 10
    # # dummy pointcloud reflectance between -1 and 1
    # lidar_r = torch.rand(size=[1000, 1]) * 2 - 1
    # lidar = torch.cat((lidar_xyz, lidar_r), dim=1).cuda()
    # voxels, indices, num_points_per_voxel = voxel_generator(lidar)

    ###
    print("Dense lidar example")
    voxel_size = [0.5, 0.5, 2.0]
    pc_range = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
  
    batch_size = 5
    input_data = []
    for i in range(batch_size):
        # dummy pointcloud xyz between -10 and 10
        lidar_xyz = torch.rand(size=[1000, 3]) * 2 - 1
        # dummy pointcloud reflectance between -1 and 1
        lidar_r = torch.rand(size=[1000, 1]) * 2 - 1
        lidar = torch.cat((lidar_xyz, lidar_r), dim=1).cuda()
        input_data.append(lidar)

    # voxels, indices, num_points_per_voxel = voxel_generator(lidar)

    # print(
    #     f"voxel shape: {voxels.shape} indices shape: {indices.shape} num_points shape: {num_points_per_voxel.shape}")
    # print(
    #     f"voxel sample: \n {voxels[0]} \n indices sample: {indices[0]} \n num_points sample: {num_points_per_voxel[0]}")
    # print(
    #     f"indices xmin: {torch.min(indices[:, 2])} indices xmax: {torch.max(indices[:, 2])}")
    # print(
    #     f"indices ymin: {torch.min(indices[:, 1])} indices ymax: {torch.max(indices[:, 1])}")
    # print(
    #     f"indices zmin: {torch.min(indices[:, 0])} indices zmax: {torch.max(indices[:, 0])}")

    pc_enc = PointCloudEncoder(voxel_size=voxel_size, pc_range=pc_range, batch_size=len(input_data))

    out = pc_enc(input_data)

    print(out.shape)
