import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_transforms(args) -> (transforms.Compose, transforms.Compose):
    """
    :param args:
    :return: transform composition for train and val
    """
    # TODO

    train_tf = transforms.Compose([
        transforms.ToTensor(),
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

    def __init__(self, args, data_path: str, transform: transforms.Compose):
        self.transform = transform

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        return torch.zeros(1), torch.ones(1)


def get_data_loaders(args) -> (DataLoader, DataLoader):
    """
    :param args:
    :return: tuple of (train data loader, val data loader)
    """
    train_tf, val_tf = get_transforms(args)

    train_dataset = KittiDataset(args, args.train_dir, train_tf)
    val_dataset = KittiDataset(args, args.val_dir, val_tf)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_data_loader_workers,
                              drop_last=True
                              )

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_data_loader_workers,
                            drop_last=True
                            )

    return train_loader, val_loader
