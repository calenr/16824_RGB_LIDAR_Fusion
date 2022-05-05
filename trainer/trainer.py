import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.metric import calc_map
import utils.utils
import wandb
from tqdm import tqdm
from utils.kitti_viewer import Calibration, draw_3d_output, draw_2d_output


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
            wandb.define_metric("train/total_loss", step_metric="train_step")
            wandb.define_metric("train/obj_conf_loss", step_metric="train_step")
            wandb.define_metric("train/noobj_conf_loss", step_metric="train_step")
            wandb.define_metric("train/coord_loss", step_metric="train_step")
            wandb.define_metric("train/shape_loss", step_metric="train_step")
            wandb.define_metric("train/angle_loss", step_metric="train_step")
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
            # return
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

            self.optimizer.zero_grad()
            outputs = self.model(images, lidars)

            obj_conf_loss, noobj_conf_loss, coord_loss, shape_loss, angle_loss = self.criterion(outputs, labels, calibs, self.args.pc_grid_size)
            loss = obj_conf_loss + noobj_conf_loss + coord_loss + shape_loss + angle_loss
            loss.backward()
            self.optimizer.step()

            if self.train_step % self.args.log_period == 0:
                # train_map = calc_map(output, label)
                train_map = torch.zeros(1)
                print(f"epoch: {epoch}, batch_idx: {batch_idx}, train_loss: {loss}, train_map: {train_map}")
                if self.args.use_wandb:
                    wandb.log({"train/total_loss": loss.item(),
                               "train/obj_conf_loss": obj_conf_loss.item(),
                               "train/noobj_conf_loss": noobj_conf_loss.item(),
                               "train/coord_loss": coord_loss.item(),
                               "train/shape_loss": shape_loss.item(),
                               "train/angle_loss": angle_loss.item(),
                               "train/map": train_map.item(),
                               "train/lr": utils.get_lr(self.optimizer),
                               "train_step": self.train_step,
                              })
            
            torch.cuda.empty_cache()

    def val_epoch(self, epoch):
        """
        One epoch of validation
        """
        self.val_step += 1
        self.model.eval()

        output_kitti_list = []
        output_list = []
        target_list = []
        calibs_list = []

        for batch_idx, batch_data in enumerate(tqdm(self.val_loader)):
            images = batch_data['image'].to(self.args.device)
            labels = batch_data['label']
            lidars = batch_data['lidar']
            calibs = batch_data['calib']

            outputs = self.model(images, lidars)
            
            for example_idx in range(outputs.shape[0]):
                output = outputs[example_idx]
                target = labels[example_idx]
                calib = calibs[example_idx]
                # TODO: add args.inference_conf_threshold
                output_kitti = self.criterion.convert_yolo_output_to_kitti_labels(output, calib, 0.0)
                output_kitti_list.append(output_kitti.to('cpu'))
                output_list.append(output.to('cpu'))
                target_list.append(target.to('cpu'))
                calibs_list.append(calib)

            torch.cuda.empty_cache()

        val_map = calc_map(output_list, target_list, self.args.MAP_overlap_threshold)

        output_stacked = torch.stack(output_list)
        obj_conf_loss, noobj_conf_loss, coord_loss, shape_loss, angle_loss = self.criterion(output_stacked, target_list, calibs_list, self.args.pc_grid_size)
        val_loss = obj_conf_loss + noobj_conf_loss + coord_loss + shape_loss + angle_loss

        if val_map > self.best_val_map:
            self.best_val_map = val_map
            self.is_best = True
        else:
            self.is_best = False

        print(f"VALIDATION epoch: {epoch}, val_loss: {val_loss}, val_map: {val_map}")
        if self.args.use_wandb:
            wandb.log({"val/loss": val_loss.item(),
                    "val/map": val_map.item(),
                    "val_step": self.val_step,
                    })

        torch.cuda.empty_cache()

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

    def visualize_sample(self):
        self.model.eval()
        for batch_idx, batch_data in enumerate(self.val_loader):
            images = batch_data['image'].to(self.args.device)
            labels = batch_data['label']
            lidars = batch_data['lidar']
            calibs = batch_data['calib']

            outputs = self.model(images, lidars)

            output_kitti = self.criterion.convert_yolo_output_to_kitti_labels(outputs[0], calibs[0], self.args.inference_conf_threshold)
            draw_3d_output(lidars[0].to('cpu').numpy(), labels[0].numpy().tolist(), calibs[0], output_kitti.detach().to('cpu').numpy().tolist())
            resizer = transforms.Resize((self.args.ori_img_h, self.args.ori_img_w), antialias=True)
            images = resizer(images)
            draw_2d_output(images[0].permute(1, 2, 0).to('cpu').numpy(), labels[0].numpy().tolist(), calibs[0], output_kitti.detach().to('cpu').numpy().tolist())
            return
