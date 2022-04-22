import argparse
import os

import torch
import numpy as np
import wandb
from data_loader.data_loader import get_data_loaders
from model.model import RgbLidarFusion
from trainer.trainer import Trainer
import multiprocessing

SEED = 420
# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def get_args(arg_list=None):
    parser = argparse.ArgumentParser(description='Hell Yeah')
    # setup params
    parser.add_argument('--train_dir', type=str, default="data/dataset_small")
    parser.add_argument('--val_dir', type=str, default="data/dataset_small")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--num_data_loader_workers', type=int, default=multiprocessing.cpu_count())
    # monitor params
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default="results/best.pth")
    parser.add_argument('--save_best_model', type=bool, default=False)
    parser.add_argument('--save_model_checkpoint', type=bool, default=False)
    parser.add_argument('--save_period', type=int, default=10)  # epoch
    parser.add_argument('--log_period', type=int, default=10)  # iteration
    parser.add_argument('--val_period', type=int, default=2)  # epoch
    parser.add_argument('--use_wandb', type=bool, default=False)
    # data params
    parser.add_argument('--image_size', type=int, default=224)
    # training params
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler_step', type=int, default=20)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    args = parser.parse_args() if str is None else parser.parse_args(arg_list)
    return args


def main(args):
    if args.use_wandb:
        wandb.login()
        wandb.init(entity="16824_rgb_lidar_fusion", project="test",
                   config={
                       "lr": args.lr,
                       "batch_size": args.batch_size,
                       "scheduler_step": args.scheduler_step,
                       "scheduler_gamma": args.scheduler_gamma,
                       "num_epochs": args.num_epochs,
                       "notes": "Write your notes about this run here"
                   })

    train_loader, val_loader = get_data_loaders(args)
    model = RgbLidarFusion(args).to(args.device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.0, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_step, args.scheduler_gamma)

    trainer = Trainer(args, model, loss_fn, optimizer, scheduler, train_loader, val_loader)
    trainer.train()

    wandb.finish()


if __name__ == '__main__':
    args = get_args()
    main(args)
