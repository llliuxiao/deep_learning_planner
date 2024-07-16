import json
import math
import os.path

import numpy as np
import torch
from einops import repeat
from torch.utils.data import DataLoader, Dataset

from src.deep_learning_planner.scripts.utils.parameters import *

dataset_root = f"/home/{os.getlogin()}/Downloads/dataset"


class RobotTransformerDataset(Dataset):
    class MetaData:
        def __init__(self):
            self.laser_path = None
            self.global_plan_path = None
            self.robot_pose = None
            self.world_size = None
            self.goal = None
            self.cmd_vel = None

    def __init__(self, mode: str):
        trajectory_list = os.listdir(dataset_root)
        join = lambda trajectory_num: os.path.join(dataset_root, trajectory_num)
        if mode == "train":
            self.trajectory_paths = list(map(join, trajectory_list[:int(len(trajectory_list) * 0.7)]))
        else:
            self.trajectory_paths = list(map(join, trajectory_list[int(len(trajectory_list) * 0.7):]))
        self.meta_data_list = []
        self._load_data()

    def _load_data(self):
        for i in range(len(self.trajectory_paths)):
            trajectory_info_file = os.path.join(self.trajectory_paths[i], "dataset_info.json")
            with open(trajectory_info_file, "r") as f:
                info = json.load(f)
            width, height, resolution = info["width"], info["height"], info["resolution"]
            trajectory_data = info["data"]
            for j, data in enumerate(trajectory_data):
                meta_data = self.MetaData()
                meta_data.world_size = (width, height)
                meta_data.goal = (-width / 2, 3.0 + 3.0 + height)
                meta_data.cmd_vel = (data["cmd_vel_linear"], data["cmd_vel_angular"])
                meta_data.robot_pose = (data["robot_x"], data["robot_y"], data["robot_yaw"])
                meta_data.laser_path = [data["laser_path"] for _ in range(laser_length)]
                for k in range(laser_length - 1, 0, -1):
                    prefix = max(0, j - k * interval)
                    meta_data.laser_path[laser_length - k - 1] = trajectory_data[prefix]["laser_path"]
                meta_data.global_plan_path = data["global_plan_path"]
                self.meta_data_list.append(meta_data)

    def __len__(self):
        return len(self.meta_data_list)

    def __getitem__(self, item):
        meta_data = self.meta_data_list[item]
        scale = torch.tensor([look_ahead_distance, look_ahead_distance], dtype=torch.float)

        # goal
        dx, dy = meta_data.goal[0] - meta_data.robot_pose[0], meta_data.goal[1] - meta_data.robot_pose[1]
        sin_yaw, cos_yaw = -math.sin(meta_data.robot_pose[2]), math.cos(meta_data.robot_pose[2])
        x = dx * cos_yaw - dy * sin_yaw
        y = dx * sin_yaw + dy * cos_yaw
        goal = torch.tensor((x, y), dtype=torch.float)
        goal = torch.div(goal, scale)

        # global plan
        global_plan = np.load(meta_data.global_plan_path)
        if len(global_plan) > 0:
            global_plan = torch.from_numpy(global_plan[:, :2]).float()
            global_plan = global_plan[:min(len(global_plan), look_ahead_poses * down_sample):down_sample, :]
            if len(global_plan) < look_ahead_poses:
                padding = repeat(goal, "d -> b d", b=look_ahead_poses - len(global_plan))
                global_plan = torch.concat([global_plan, padding])
        else:
            global_plan = repeat(goal, "d -> b d", b=look_ahead_poses)
        global_plan = torch.div(global_plan, scale)

        # scan
        laser = np.array([np.load(path) for path in meta_data.laser_path]) / laser_range
        laser = torch.tensor(laser, dtype=torch.float)
        return laser, goal, global_plan


def load_data(mode: str, batch_size=128):
    data = RobotTransformerDataset(mode)
    return DataLoader(dataset=data,
                      batch_size=batch_size,
                      # collate_fn=collate_fn,
                      shuffle=False,
                      num_workers=24)


if __name__ == "__main__":
    dataset = RobotTransformerDataset("train")
