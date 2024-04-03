# torch
import torch
import torch.nn as nn

# utils
import math
import os

# visualization
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data import load_data
from transformer_network import RobotTransformer

linux_user = os.getlogin()
save_root = f"/home/{linux_user}/isaac_sim_ws/src/supervised_learning_planner/transformer_logs"
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


class Trainner:
    def __init__(self, batch_size=128, lr=1e-5, device=torch.device("cuda"), frame=6):
        self.frame = frame
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model = RobotTransformer()
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

    def resume_checkpoint(self, path):
        checkpoint_path = path
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        self.lr_decay.last_epoch = start_epoch

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
        # 定义mask上三角矩阵(不含对角线）
        laser_mask = torch.ones((self.frame, self.frame), dtype=torch.bool).triu(1)
        laser_mask = torch.stack([laser_mask for _ in range(torch.cuda.device_count())]).to(self.device)
        with tqdm(total=len(self.train_data_loader) * self.batch_size, desc=f"training_epoch{num}") as pbar:
            for j, (laser, global_plan, goal, cmd_vel) in enumerate(self.train_data_loader):
                cmd_vel = cmd_vel.to(self.device)
                laser = laser.to(self.device)
                global_plan = global_plan.to(self.device)
                goal = goal.to(self.device)
                predict = self.model(laser, global_plan, goal, laser_mask)
                loss = self.loss_fn(predict, cmd_vel)

                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # visualization
                batch_length = laser.shape[0]
                self.training_total_step += batch_length
                epoch_steps += batch_length
                epoch_loss += batch_length * loss.item()
                pbar.update(batch_length)
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
        laser_mask = torch.ones((self.frame, self.frame), dtype=torch.bool).triu(1)
        laser_mask = torch.stack([laser_mask for _ in range(torch.cuda.device_count())]).to(self.device)
        with torch.no_grad():
            with tqdm(total=len(self.eval_data_loader) * self.batch_size, desc=f"evaluating_epoch{num}") as pbar:
                for j, (laser, global_plan, goal, cmd_vel) in enumerate(self.eval_data_loader):
                    cmd_vel = cmd_vel.to(self.device)
                    laser = laser.to(self.device)
                    global_plan = global_plan.to(self.device)
                    goal = goal.to(self.device)
                    predict = self.model(laser, global_plan, goal, laser_mask)
                    loss = self.loss_fn(predict, cmd_vel)

                    # visualization
                    batch_length = laser.shape[0]
                    pbar.update(batch_length)
                    epoch_steps += batch_length
                    epoch_loss += batch_length * loss.item()
                    self.eval_total_step += batch_length
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


# best_model = "/home/gr-agv-lx91/isaac_sim_ws/src/supervised_learning_planner/transformer_logs/model1/best.pth"
if __name__ == "__main__":
    planner = Trainner()
    # planner.resume_checkpoint(best_model)
    epoch = 200
    for i in range(epoch):
        planner.train(i)
        planner.eval(i)
        # planner.save_checkpoint(i)
        if i > 0 and i % 5 == 0:
            planner.lr_decay.step()
    planner.train_summary_writer.close()
    planner.eval_summary_writer.close()
