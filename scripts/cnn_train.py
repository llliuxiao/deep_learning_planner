# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from cnn_network import CNNModel

# utils
import json
import math
import os
import numpy as np

# visualization
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

linux_user = os.getlogin()
train_dataset_root = f"/home/{linux_user}/Downloads/navigation_pretraining_dataset/pretraining_dataset/office"
eval_dataset_root = f"/home/{linux_user}/Downloads/navigation_pretraining_dataset/pretraining_dataset/office"

ignore_init_step_num = 0
ignore_end_step_num = 0
subsample_interval = 1
save_root = f"/home/{linux_user}/isaac_sim_ws/src/deep_learning_planner/logs"
index = 0
while True:
    if os.path.exists(f"{save_root}/model{index}"):
        index += 1
        continue
    model_save_dir = f"{save_root}/model{index}"
    os.mkdir(model_save_dir)
    train_save_dir = f"{save_root}/runs/train{index}"
    eval_save_dir = f"{save_root}/runs/eval{index}"
    break


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
        (goal, cmd_vel, laser_path) = self.data[item]
        # the max range of laser is 10m. so here normalize it to [0, 1]
        laser = np.load(laser_path) / 10.0
        goal = np.array(goal) / np.array([10.0, 10.0, math.pi])
        return (torch.concat([torch.from_numpy(laser).float(), torch.from_numpy(goal).float()]),
                torch.Tensor(cmd_vel).float())

    def _subsample(self):
        for (_, trajectory) in self.raw_data.items():
            length = len(trajectory["data"])
            for num in range(ignore_init_step_num, length - ignore_end_step_num, subsample_interval):
                step = trajectory["data"][num]
                target = (step["target_x"], step["target_y"], step["target_yaw"])
                cmd_vel = (step["cmd_vel_linear"], step["cmd_vel_angular"])
                laser_path = step["laser_path"]
                self.data.append((target, cmd_vel, laser_path))


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
        self.model = CNNModel()
        self.model = nn.DataParallel(self.model).to(self.device)
        # self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = VelocityCmdLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.00001)
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

        self.train_data_loader = load_data("train", batch_size)
        self.eval_data_loader = load_data("eval", batch_size)

        self.train_summary_writer = SummaryWriter(train_save_dir)
        self.eval_summary_writer = SummaryWriter(eval_save_dir)

        self.training_total_step = 0
        self.eval_total_step = 0

        self.best_loss = math.inf
        torch.manual_seed(1)
        print(self.model)

    def get_single_loss(self, predicts, targets):
        sub = torch.sub(targets, predicts)
        fabs = torch.abs(sub)
        sum = torch.sum(fabs, dim=0)
        squeeze = torch.squeeze(sum)
        divide = torch.divide(squeeze, torch.tensor(len(predicts), dtype=torch.float))
        return divide[0].item(), divide[1].item()


    def train(self, num):
        self.model.train(True)
        epoch_loss = torch.tensor(0.0, dtype=torch.float)
        epoch_steps = 0
        with tqdm(total=len(self.train_data_loader), desc=f"training_epoch{num}") as pbar:
            for j, (data, cmd_vel) in enumerate(self.train_data_loader):
                cmd_vel = cmd_vel.to(self.device)
                data = torch.reshape(data, (-1, 1, 1083))
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
                    self.train_summary_writer.add_scalar("step_loss", loss.item(), self.training_total_step)
                    linear, angular = self.get_single_loss(predict, cmd_vel)
                    self.train_summary_writer.add_scalar("training_linear_loss", linear, self.training_total_step)
                    self.train_summary_writer.add_scalar("training_angular_loss", angular, self.training_total_step)
        self.train_summary_writer.add_scalar("learning_rate", self.lr_decay.get_lr()[0], num)
        self.train_summary_writer.add_scalar("epoch_loss", epoch_loss.item() / epoch_steps, num)

    def eval(self, num):
        self.model.train(False)
        epoch_loss = torch.tensor(0.0, dtype=torch.float)
        epoch_steps = 0
        with torch.no_grad and tqdm(total=len(self.eval_data_loader), desc=f"evaluating_epoch{num}") as pbar:
            for j, (data, cmd_vel) in enumerate(self.eval_data_loader):
                cmd_vel = cmd_vel.to(self.device)
                data = torch.reshape(data, (-1, 1, 1083))
                predict = self.model(data)
                loss = self.loss_fn(predict, cmd_vel)
                pbar.update(len(data))
                epoch_steps += len(data)
                epoch_loss += len(data) * loss.item()
                self.eval_total_step += len(data)
                if j % 50 == 0:
                    self.eval_summary_writer.add_scalar("step_loss", loss.item(), self.eval_total_step)
        self.eval_summary_writer.add_scalar("epoch_loss", epoch_loss.item() / epoch_steps, num)
        self.save_best_model(epoch_loss.item() / epoch_steps, num)

    def save_checkpoint(self, num):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": num
        }
        torch.save(checkpoint, f"{model_save_dir}/epoch{num}.pth")

    def save_best_model(self, loss, num):
        if loss < self.best_loss:
            self.best_loss = loss
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": num
            }
            torch.save(checkpoint, f"{model_save_dir}/best.pth")


if __name__ == "__main__":
    planner = DeepMotionPlannerTrainner()
    epoch = 200
    for i in range(epoch):
        planner.train(i)
        planner.eval(i)
        # planner.save_checkpoint(i)
        if i > 0 and i % 3 == 0:
            planner.lr_decay.step()
    planner.train_summary_writer.close()
    planner.eval_summary_writer.close()
