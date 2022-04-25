import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.metric import calc_map
import utils.utils
import wandb
from tqdm import tqdm
from utils.kitti_viewer import draw_3d_output, draw_2d_output


class Trainer:
    def __init__(self, args, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler, train_loader: DataLoader, val_loader: DataLoader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.start_epoch = 1
        self.train_step = 1
        self.val_step = 1

        self.best_val_map = 0
        self.is_best = False

        if self.args.load_checkpoint:
            self.load_model()

        if args.use_wandb:
            wandb.define_metric("train_step")
            wandb.define_metric("val_step")
            wandb.define_metric("train/loss", step_metric="train_step")
            wandb.define_metric("train/map", step_metric="train_step")
            wandb.define_metric("train/lr", step_metric="train_step")
            wandb.define_metric("val/loss", step_metric="val_step")
            wandb.define_metric("val/map", step_metric="val_step")

    def train(self):
        """
        run the training sequence, validate and save models along the way
        """
        for epoch in range(self.start_epoch, self.args.num_epochs + 1):
            self.train_epoch(epoch)
            self.scheduler.step()

            if epoch % self.args.val_period == 0:
                self.val_epoch(epoch)

            if self.args.save_best_model and self.is_best:
                self.save_model(epoch, str("results/checkpoint_best.pth"))

            if self.args.save_model_checkpoint and epoch % self.args.save_period == 0:
                self.save_model(epoch, str("results/checkpoint_epoch{}.pth".format(epoch)))

    def train_epoch(self, epoch: int):
        """
        One epoch of training
        """
        self.model.train()
        for batch_idx, batch_data in enumerate(self.train_loader):
            self.train_step += 1
            images = batch_data['image'].to(self.args.device)
            labels = batch_data['label']
            lidars = batch_data['lidar']
            calibs = batch_data['calib']
            # draw_3d_output(lidars[0].to('cpu').numpy(), labels[0].numpy().tolist(), calibs[0])

            self.optimizer.zero_grad()
            output = self.model(images, lidars)
            loss = self.criterion(output, labels, calibs, self.args.pc_grid_size)
            loss.backward()
            self.optimizer.step()

            if self.train_step % self.args.log_period == 0:
                # train_map = calc_map(output, label)
                train_map = torch.zero(1)
                print(f"epoch: {epoch}, batch_idx: {batch_idx}, train_loss: {loss}, train_map: {train_map}")
                wandb.log({"train/loss": loss.item(),
                           "train/map": train_map.item(),
                           "train/lr": utils.get_lr(self.optimizer),
                           "train_step": self.train_step,
                           })

    def val_epoch(self, epoch):
        """
        One epoch of validation
        """
        self.val_step += 1
        self.model.eval()

        output_agg = []
        target_agg = []

        for batch_idx, batch_data in enumerate(tqdm(self.val_loader)):
            data, target = batch_data
            data, target = data.to(self.args.device), target.to(self.args.device)

            output = self.model(data)

            output_agg.append(output)
            target_agg.append(target)

        output_agg = torch.vstack(output_agg)
        target_agg = torch.vstack(target_agg)

        # val_map = calc_map(output_agg, target_agg)
        # val_loss = self.criterion(output_agg, target_agg)
        val_map = torch.zero(1)
        val_loss = torch.zero(1)

        if val_map > self.best_val_map:
            self.best_val_map = val_map
            self.is_best = True
        else:
            self.is_best = False

        print(f"VALIDATION epoch: {epoch}, val_loss: {val_loss}, val_map: {val_map}")
        wandb.log({"val/loss": val_loss.item(),
                   "val/map": val_map.item(),
                   "val_step": self.val_step,
                   })

    def save_model(self, epoch, filename):
        """
        Save the model snapshot
        :param epoch: current epoch
        :param filename: path filename to save to
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args,
            'val_map': self.best_val_map
        }

        torch.save(state, filename)

    def load_model(self):
        """
        Load model from checkpoint, path is taken from args
        """
        checkpoint = torch.load(self.args.checkpoint_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_map = checkpoint['val_map']

        curr_arch = type(self.model).__name__
        if curr_arch != checkpoint['arch']:
            print("Loading checkpoint warning, model is not the same architecture")
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Model checkpoint loaded")
