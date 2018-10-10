import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ACNet(nn.Module):

    def __init__(self, available_actions_count):
        super(ACNet, self).__init__()
        # if 64*64*3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        # out is (15*15*32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # out is (6*6*64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1)
        # out is (3*3*64)

        self.fc1 = nn.Linear(576, 512)
        self.action_head = nn.Linear(512, available_actions_count)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        return F.softmax(action_scores, dim=-1),   state_values
