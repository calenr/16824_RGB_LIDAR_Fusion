import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from os import path as osp
from PIL import Image
from utils.kitti_viewer import Calibration

CLASSNAMES_TO_IDX = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2
}

def collate_fn(batch):
    image = [item['image'] for item in batch]
    label = [item['label'] for item in batch]
    lidar = [item['lidar'] for item in batch]
    calib = [item['calib'] for item in batch]

    image = torch.stack(image)

    data = dict()
    data['image'] = image
    data['label'] = label
    data['lidar'] = lidar
    data['calib'] = calib

    return data


def get_transforms(args) -> tuple[transforms.Compose, transforms.Compose]:
    """
    :param args:
    :return: transform composition for train and val
    """
    # TODO

    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size), antialias=True)
    ])

    val_tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_tf, val_tf


class KittiDataset(Dataset):
    """
    DataSet class to that holds and returns the KITTI data
    """

    # TODO

    def __init__(self, args, data_path: str, transform: transforms.Compose = None,
                 training: bool = True):
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
        self.data_path = data_path
        self.training = training
        # split = "training" if self.training else "testing"
        split = "training"

        # Get image file paths
        self.image_data_path = osp.join(data_path, "data_object_image_2", split, "image_2")
        assert osp.exists(self.image_data_path)
        self.image_files = sorted(os.listdir(self.image_data_path))
        print(f"Num images files: {len(self.image_files)}")

        # Get label file paths
        if training:
            self.label_data_path = osp.join(data_path, "data_object_label_2", split, "label_2")
            assert osp.exists(self.label_data_path)
            self.label_files = sorted(os.listdir(self.label_data_path))
            assert len(self.label_files) == len(self.image_files)
            print(f"Num label files: {len(self.label_files)}")
        else:
            self.label_data_path = None
            self.label_files = None

        # Get velodyne file paths
        self.velodyne_data_path = osp.join(data_path, "data_object_velodyne", split, "velodyne")
        assert osp.exists(self.velodyne_data_path)
        self.velodyne_files = sorted(os.listdir(self.velodyne_data_path))
        assert len(self.velodyne_files) == len(self.image_files)
        print(f"Num velodyne files: {len(self.velodyne_files)}")

        # Get calibration file paths
        self.calibration_data_path = osp.join(data_path, "data_object_calib", split, "calib")
        assert osp.exists(self.calibration_data_path)
        self.calibration_files = sorted(os.listdir(self.calibration_data_path))
        assert len(self.calibration_files) == len(self.image_files)
        print(f"Num calibration files: {len(self.calibration_files)}")

        self.len = len(self.image_files)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Load image
        image_path = osp.join(self.image_data_path, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        label_list = []

        # Load Labels
        if self.training:
            label_path = osp.join(self.label_data_path, self.label_files[idx])
            labels = [line.rstrip() for line in open(label_path)]
            for label in labels:
                data = label.split(' ')
                data[1:] = [float(x) for x in data[1:]]
                if data[0] in CLASSNAMES_TO_IDX.keys():
                    data[0] = CLASSNAMES_TO_IDX[data[0]]
                else:
                    data[0] = len(CLASSNAMES_TO_IDX.items())

                label_list.append(torch.Tensor(data))
        else:
            label_tensor = torch.zeros(1)

        label_tensor = torch.stack(label_list)

        velo_path = osp.join(self.velodyne_data_path, self.velodyne_files[idx])
        velo_np = np.fromfile(velo_path, dtype=np.float32)
        velo_np = velo_np.reshape(-1, 4)
        velo = torch.from_numpy(velo_np)

        calib_path = osp.join(self.calibration_data_path, self.calibration_files[idx])
        calib = Calibration(calib_path)

        data = dict()
        data['image'] = image
        data['label'] = label_tensor
        data['lidar'] = velo
        data['calib'] = calib

        return data


def get_data_loaders(args) -> tuple[DataLoader, DataLoader]:
    """
    :param args:
    :return: tuple of (train data loader, val data loader)
    """
    train_tf, val_tf = get_transforms(args)

    train_dataset = KittiDataset(args, args.train_dir, train_tf, training=True)
    val_dataset = KittiDataset(args, args.val_dir, val_tf, training=False)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_data_loader_workers,
                              drop_last=True,
                              collate_fn=collate_fn,
                              )

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_data_loader_workers,
                            drop_last=True,
                            collate_fn=collate_fn,
                            )

    return train_loader, val_loader
