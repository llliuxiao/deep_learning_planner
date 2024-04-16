import json
import os.path

import numpy as np
import pandas as pd
import torch
from einops import repeat
from torch.utils.data import DataLoader, Dataset
from parameters import *

dataset_root = "/home/gr-agv-lx91/Downloads/pretraining_dataset"
scenes = ["hospital", "office"]
train_dataset_info_path = os.path.join(dataset_root, "train_dataset.csv")
eval_dataset_info_path = os.path.join(dataset_root, "eval_dataset.csv")


class RobotTransformerDataset(Dataset):
    class MetaData:
        def __init__(self):
            self.laser_path = None
            self.global_plan_path = None
            self.goal = None
            self.cmd_vel = None

    def __init__(self, mode: str):
        self.dataset_info_path = train_dataset_info_path if mode == "train" else eval_dataset_info_path
        self.dataframe = pd.read_csv(self.dataset_info_path)
        self.data_info = []
        with open(f"{dataset_root}/hospital/dataset_info.json") as f1:
            with open(f"{dataset_root}/office/dataset_info.json") as f2:
                self.hospital = json.load(f1)
                self.office = json.load(f2)
        self._load_data()

    def _load_data(self):
        for data in self.dataframe.values:
            scene, trajectory_prefix, step_prefix = data
            trajectory = f"trajectory{trajectory_prefix}"
            if scene == "hospital":
                step_data = self.hospital[trajectory]["data"]
                self._append_meta_data(step_data, step_prefix)
            elif scene == "office":
                step_data = self.office[trajectory]["data"]
                self._append_meta_data(step_data, step_prefix)

    def _append_meta_data(self, step_data, step_prefix):
        meta_data = self.MetaData()
        meta_data.global_plan_path = step_data[step_prefix]["global_plan_path"]
        meta_data.goal = (step_data[step_prefix]["target_x"],
                          step_data[step_prefix]["target_y"],
                          step_data[step_prefix]["target_yaw"])
        meta_data.cmd_vel = (step_data[step_prefix]["cmd_vel_linear"], step_data[step_prefix]["cmd_vel_angular"])
        meta_data.laser_path = [step_data[step_prefix]["laser_path"] for _ in range(laser_length)]
        for i in range(laser_length - 1, 0, -1):
            prefix = step_prefix - i * interval
            if prefix < 0:
                continue
            else:
                meta_data.laser_path[laser_length - i - 1] = step_data[prefix]["laser_path"]
        self.data_info.append(meta_data)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, item):
        meta_data = self.data_info[item]
        scale = torch.tensor([look_ahead_distance, look_ahead_distance, torch.pi], dtype=torch.float)
        goal = torch.tensor(meta_data.goal, dtype=torch.float)
        goal = torch.div(goal, scale)
        cmd_vel = torch.tensor(meta_data.cmd_vel, dtype=torch.float)
        global_plan = torch.from_numpy(np.load(meta_data.global_plan_path)).float()
        if len(global_plan) > 0:
            global_plan = global_plan[:min(len(global_plan), look_ahead_poses * down_sample):down_sample, :]
            if len(global_plan) < look_ahead_poses:
                padding = repeat(goal, "d -> b d", b=look_ahead_poses - len(global_plan))
                global_plan = torch.concat([global_plan, padding])
        else:
            global_plan = repeat(goal, "d -> b d", b=look_ahead_poses)
        global_plan = torch.div(global_plan, scale)
        laser = np.array([np.load(path) for path in meta_data.laser_path]) / laser_range
        laser = torch.tensor(laser, dtype=torch.float)
        return laser, global_plan, goal, cmd_vel


# def collate_fn(batch):
#     laser = torch.stack([item[0] for item in batch])
#     global_plan = [item[1] for item in batch]
#     global_plan = pad_sequence(global_plan, batch_first=True)
#     goal = torch.stack([item[2] for item in batch])
#     cmd_vel = torch.stack([item[3] for item in batch])
#     return laser, global_plan, goal, cmd_vel


def load_data(mode: str, batch_size=128):
    data = RobotTransformerDataset(mode)
    return DataLoader(dataset=data,
                      batch_size=batch_size,
                      # collate_fn=collate_fn,
                      shuffle=False,
                      num_workers=24)


if __name__ == "__main__":
    dataset = RobotTransformerDataset("train")
