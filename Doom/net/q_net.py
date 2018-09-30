# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        # if 30*45
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        # out is (8,9,14)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        # out is (8,4,6)
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return self.fc2(x)