# torch
# utils
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
# visualization
from tqdm import tqdm

from network import Model

linux_user = os.getlogin()
train_dataset_root = f"/home/{linux_user}/Downloads/pretraining_dataset"
eval_dataset_root = f"/home/{linux_user}/Downloads/pretraining_dataset_eval"

ignore_init_step_num = 50
ignore_end_step_num = 100
subsample_interval = 4
save_root = f"/home/{linux_user}/isaac_sim_ws/src/supervised_learning_planner/logs"


class RobotDataset(Dataset):
    def __init__(self, mode):
        if mode == "train":
            with open(os.path.join(train_dataset_root, "dataset_info.json"), "r") as file:
                self.raw_data = json.load(file)
        else:
            with open(os.path.join(eval_dataset_root, "dataset_info.json"), "r") as file:
                self.raw_data = json.load(file)
        self.data = []
        self._subsample()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        (goal, cmd_vel, last_cmd_vel, laser_path) = self.data[item]
        laser = np.load(laser_path)
        return (torch.concat([torch.from_numpy(laser).float(), torch.Tensor(goal).float(),
                              torch.Tensor(last_cmd_vel).float()]), torch.Tensor(cmd_vel).float())

    def _subsample(self):
        for (_, trajectory) in self.raw_data.items():
            length = len(trajectory["data"])
            for num in range(ignore_init_step_num, length - ignore_end_step_num, subsample_interval):
                step = trajectory["data"][num]
                last_step = trajectory["data"][num - 1]
                target = (step["target_x"], step["target_y"], step["target_yaw"])
                cmd_vel = (step["cmd_vel_linear"], step["cmd_vel_angular"])
                last_cmd_vel = (last_step["cmd_vel_linear"], last_step["cmd_vel_angular"])
                laser_path = step["laser_path"]
                self.data.append((target, cmd_vel, last_cmd_vel, laser_path))


def load_data(mode, batch_size):
    dataset = RobotDataset(mode)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=24)


class VelocityCmdLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, inputs, targets):
        sub = torch.sub(targets, inputs)
        pow = torch.pow(sub, 2)
        sum = torch.sum(pow, dim=1)
        sqrt = torch.sqrt(sum)
        return torch.divide(torch.sum(sqrt), torch.tensor(len(inputs), dtype=torch.float))


class DeepMotionPlannerTrainner:
    def __init__(self, batch_size=128, lr=1e-3, device=torch.device("cuda")):
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model = Model()
        self.model = nn.DataParallel(self.model).to(self.device)
        # self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = VelocityCmdLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

        self.train_data_loader = load_data("train", batch_size)
        self.eval_data_loader = load_data("eval", batch_size)

        self.summary_writer = SummaryWriter(f"{save_root}/runs")

        self.training_total_step = 0
        self.eval_total_step = 0

    def train(self, num):
        self.model.train(True)
        epoch_loss = torch.tensor(0.0, dtype=torch.float)
        epoch_steps = 0
        with tqdm(total=len(self.train_data_loader), desc=f"training_epoch{num}") as pbar:
            for j, (data, cmd_vel) in enumerate(self.train_data_loader):
                cmd_vel = cmd_vel.to(self.device)
                data = torch.reshape(data, (-1, 1, 905))
                predict = self.model(data)
                loss = self.loss_fn(predict, cmd_vel)

                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # visualization
                self.training_total_step += len(data)
                epoch_steps += len(data)
                epoch_loss += len(data) * loss.item()
                pbar.update(len(data))
                if j % 50 == 0:
                    self.summary_writer.add_scalar("training_loss", loss.item(), self.training_total_step)
        self.summary_writer.add_scalar("learning_rate", self.lr_decay.get_lr()[0], num)
        self.summary_writer.add_scalar("train_epoch_loss", epoch_loss.item() / epoch_steps, num)

    def eval(self, num):
        self.model.train(False)
        epoch_loss = torch.tensor(0.0, dtype=torch.float)
        epoch_steps = 0
        with torch.no_grad and tqdm(total=len(self.eval_data_loader), desc=f"evaluating_epoch{num}") as pbar:
            for j, (data, cmd_vel) in enumerate(self.eval_data_loader):
                cmd_vel = cmd_vel.to(self.device)
                data = torch.reshape(data, (-1, 1, 905))
                predict = self.model(data)
                loss = self.loss_fn(predict, cmd_vel)
                pbar.update(len(data))
                epoch_steps += len(data)
                epoch_loss += len(data) * loss.item()
                self.eval_total_step += len(data)
                if j % 50 == 0:
                    self.summary_writer.add_scalar("eval_loss", loss.item(), self.eval_total_step)
        self.summary_writer.add_scalar("eval_epoch_loss", epoch_loss.item() / epoch_steps, num)

    def save_checkpoint(self, num):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": num
        }
        torch.save(checkpoint, f"{save_root}/epoch{num}.pth")


if __name__ == "__main__":
    planner = DeepMotionPlannerTrainner()
    epoch = 100
    for i in range(epoch):
        planner.train(i)
        planner.eval(i)
        planner.save_checkpoint(i)
        if i > 0 and i % 2 == 0:
            planner.lr_decay.step()
    planner.summary_writer.close()
