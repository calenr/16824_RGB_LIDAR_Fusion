# RGB LiDAR Fusion

## Description

## Installation

- Tested on Ubuntu 18.04 with RTX3070 and CUDA 11.3

```commandline
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install scikit-image scipy numba pillow matplotlib flask pyqtgraph pyopengl
pip install fire tensorboardX protobuf opencv-python spconv-cu113 tqdm wandb
```

## Implementation

## Results


## Kelvin's Notes

Kitti original img size: `(3, 375, 1242)`
res18 output on this img size: `(512, 12, 39)`

Questions
- What are the shape and data of features, num_voxels, coors
- Where do we do the pillar binning
- How to convert (C, P) feature back to (C, H ,W)

