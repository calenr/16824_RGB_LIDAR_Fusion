# RGB LiDAR Fusion
https://calenr.github.io/16824_RGB_LIDAR_Fusion/
## Description

End-to-end PyTorch model to detect cars in 3D by fusing RGB and LiDAR features. Trained and tested on KITTI dataset.

## Installation

- Tested on Ubuntu 18.04 and CUDA 11.3

```
conda env create -f environment.yml
```

OR

```commandline
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install scikit-image scipy numba pillow matplotlib flask pyqtgraph pyopengl
pip install fire tensorboardX protobuf opencv-python spconv-cu113 tqdm wandb scikit-learn
```
