import torch
import torch.nn as nn
from spconv.utils import Point2VoxelGPU3d


def get_voxel_gen(vsize_xyz: list[float] = [0.16, 0.16, 4.0],
                  coors_range_xyz: list[float] = [0, -32.0, -3, 52.8, 32.0, 1], num_point_features: int = 4,
                  max_num_voxels: int = 20000,
                  max_num_points_per_voxel: int = 5):
    """
    :param vsize_xyz: voxel size in x,y,z e.g. [0.16, 0.16, 4.0]
    :param coors_range_xyz: point cloud range values [xmin, ymin, zmin, xmax, ymax, zmax]
    :param num_point_features: dimension of the pointcloud data, 4 if xyzr, 3 if xyz
    :param max_num_voxels:
    :param max_num_points_per_voxel:
    :return:
    """
    return Point2VoxelGPU3d(
        vsize_xyz=vsize_xyz, coors_range_xyz=coors_range_xyz,
        num_point_features=num_point_features, max_num_voxels=max_num_voxels,
        max_num_points_per_voxel=max_num_points_per_voxel
    )


class Passthrough(nn.Module):
    """
    Does nothing, just passthrough the input
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
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

    def forward(self, inputs):

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = self.relu(x)

        # x should be (C, P, N) shape here

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        # x should be (C, P) shape here

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
                 num_filters=(64, 128, 256,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
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

        super().__init__()
        self.name = 'PillarFeatureNet'
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

    def forward(self, features, num_voxels, coors):
        """
        TODO: what are these??
        :param features: My guess is that it is pointcloud [batch, Num of pts, 4] (x, y, z, r)
        :param num_voxels: [num_voxels]
        :param coors: [num_voxels, 4], my guess is that it is YX btm left and YX top right coordinates of
        the pillar that a particular point belongs to
        :return:
        """

        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        # Calculate Xc, Yc, Zc, offset from arithmetic mean
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        # Calculate Xp, x offset from pillar center
        f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        # Calculate Yp, y offset from pillar center
        f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # My guess of shape is
        # [batch, num_pts, N=, D=9]
        # [batch, num_voxels, max_num_points_per_voxel, 7]?????
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        # output should be (C, P) shape here
        return features.squeeze()


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


if __name__ == '__main__':
    test = PillarFeatureNet()
