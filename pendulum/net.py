import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import gym

S_DIM, A_DIM = 3, 1


class critic(nn.Module):
    def __init__(self):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(S_DIM, 100)
        self.v = nn.Linear(100, 1)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = self.v(x)

        return x

    def cal_adv(self, disocut_r, v):
        return disocut_r - v


class actor(nn.Module):
    def __init__(self):
        super(actor, self).__init__()
        self.l1 = nn.Linear(S_DIM, 100)
        self.l2 = nn.Linear(100, A_DIM)
        
    def forward(self, s):
        x = F.relu(self.l1(s))
        mu = 2 * F.tanh(self.l2(x))
        sigma = F.softplus(self.l2(x))
        norm_dist = Normal(loc=mu, scale=sigma)

        return norm_dist

