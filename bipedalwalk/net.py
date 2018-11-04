import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""
    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 *\
        np.log(2 * np.pi) - log_std
    log_density = log_density.sum(dim=1, keepdim=True)
    return log_density


class MLPPolicy(nn.Module):
    def __init__(self, obs_space, action_space):
        super(MLPPolicy, self).__init__()
        # action network
        self.act_fc1 = nn.Linear(obs_space, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.mu = nn.Linear(128, action_space)
        self.mu.weight.data.mul_(0.1)
        # torch.log(std)
        self.logstd = nn.Parameter(torch.zeros(action_space))

        # value network
        self.value_fc1 = nn.Linear(obs_space, 64)
        self.value_fc2 = nn.Linear(64, 128)
        self.value_fc3 = nn.Linear(128, 1)
        self.value_fc3.weight.data.mul(0.1)

    def forward(self, x):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        act = self.act_fc1(x)
        act = F.tanh(act)
        act = self.act_fc2(act)
        act = F.tanh(act)
        mean = self.mu(act)  # N, num_actions
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)
       
        

        # value
        v = self.value_fc1(x)
        v = F.tanh(v)
        v = self.value_fc2(v)
        v = F.tanh(v)
        v = self.value_fc3(v)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, action):
        v, _, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


'''
v =  tensor([[-0.1048],
        [-0.1362]], grad_fn=<ThAddmmBackward>)
tensor([[ 1.1269,  0.6877],
        [ 0.2316, -1.0929]], grad_fn=<NormalBackward3>)
logprob =  tensor([[-2.6890],
        [-2.4414]], grad_fn=<SumBackward1>)
mean =  tensor([[ 0.0370, -0.0295],
        [ 0.0260, -0.0136]], grad_fn=<ThAddmmBackward>)
Parameter containing:
tensor([0., 0.], requires_grad=True)

'''