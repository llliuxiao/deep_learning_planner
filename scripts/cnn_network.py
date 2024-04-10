# The neural network architecture is from paper in ICRA 2017:
# From Perception to Decision: A Data-driven Approach to End-to-end Motion Planning for Autonomous Ground Robots

import torch
import torch.nn as nn


class BaseConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.conv(tensor)


# width = (width + 2 * padding - kernel_size) / stride + 1
class CNNModel(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(
            BaseConvolutionBlock(1, 64, 7, 3),
            nn.MaxPool1d(kernel_size=3, stride=2)  #
        )
        self.conv2 = nn.Sequential(
            BaseConvolutionBlock(64, 64, 3, 1, 1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(),
            BaseConvolutionBlock(64, 64, 3, 1, 1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64)
        )
        self.avg_pool = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2)
        )
        self.mlp = nn.Sequential(
            nn.Linear(5635, 1024), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=dropout_rate),
        )
        self.actor = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(p=dropout_rate),
        )
        self.action_net = nn.Linear(512, 2)
        self.critic = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(p=dropout_rate),
        )
        self.value_net = nn.Linear(256, 1)

    def forward(self, tensor):
        laser, goal = tensor[:, :, :1080], tensor[:, :, 1080:1083]
        goal = torch.reshape(goal, (-1, 3))
        conv1 = self.conv1(laser)
        conv2 = self.conv2(conv1)
        res1 = torch.add(conv1, conv2)
        conv3 = self.conv3(res1)
        res2 = torch.add(conv2, conv3)
        avg_pool = self.avg_pool(res2)
        avg_pool = torch.reshape(avg_pool, (-1, 5632))
        concat = torch.concat((avg_pool, goal), dim=1)
        return self.action_net(self.actor(self.mlp(concat)))
