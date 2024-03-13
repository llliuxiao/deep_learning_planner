# The neural network architecture is from paper in ICRA 2017:
# From Perception to Decision: A Data-driven Approach to End-to-end Motion Planning for Autonomous Ground Robots

import torch
import torch.nn as nn


class BaseConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.conv(tensor)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            BaseConvolutionBlock(1, 64, 7, 3),
            nn.MaxPool1d(kernel_size=3)
        )
        self.conv2 = nn.Sequential(
            BaseConvolutionBlock(64, 64, 3, 1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=64)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(),
            BaseConvolutionBlock(64, 64, 3, 1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=64)
        )
        self.avg_pool = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3)
        )
        self.mlp = nn.Sequential(
            nn.Linear(6083, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, tensor):
        laser, goal, last_cmd_vel = tensor[:, :, :900], tensor[:, :, 900:903], tensor[:, :, 903:]
        goal = torch.reshape(goal, (-1, 3))
        conv1 = self.conv1(laser)
        conv2 = self.conv2(conv1)
        value = torch.concat([conv1, conv2], dim=2)
        conv3 = self.conv3(value)
        value = torch.concat([conv2, conv3], dim=2)
        avg_pool = self.avg_pool(value)
        avg_pool = torch.reshape(avg_pool, (-1, 6080))
        value = torch.concat((avg_pool, goal), dim=1)
        return self.mlp(value)
