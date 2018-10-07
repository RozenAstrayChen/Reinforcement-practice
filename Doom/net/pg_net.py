# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):

    def __init__(self, available_actions_count):
        super(Net, self).__init__()

        # if 1*60*108
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        # out is (14,26,32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # out is (6, 12, 64)

        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 4608)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x
