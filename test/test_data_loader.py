from utils import utils
from data_loader.data_loader import get_data_loaders
from main import get_args


def main(args):
    train_loader, val_loader = get_data_loaders(args)
    print("Attempting to utilize train_loader")
    print(f"Num batchs in train leader: {len(train_loader)}")
    for data in train_loader:

        break
    print("Attempting to utilize val_loader")
    print(f"Num batchs in train leader: {len(val_loader)}")
    for data in val_loader:
        break

if __name__ == "__main__":
    arg_list = ["--train_dir",
                "/home/ubuntu/project_efs/dataset_small",
                "--val_dir",
                "/home/ubuntu/project_efs/dataset_small"]
    args = get_args(arg_list)
    main(args)
