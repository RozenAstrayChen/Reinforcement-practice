import math
import torch
import torch.nn as nn
import numpy as np
#from torch.distributions import Categorical
from torch.autograd import Variable
from distrbutions import Categorical
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
        self.fc1 = nn.Linear(obs_space, 32)
        self.fc2 = nn.Linear(32, 32)
        self.dist = Categorical(32, action_space)
        self.critic = nn.Linear(32, 1)
        

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        v = self.critic(x)
        
        return v, x
    
    def action(self, s, deterministic=False):
        value, x = self.forward(s)
        action = self.dist.sample(x, deterministic)
        
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)

        return value, action, action_log_probs 

    def evaluate_actions(self, s, actions):
        actions  = torch.autograd.Variable(actions).long()
        value, x = self.forward(s)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        
        return value, action_log_probs, dist_entropy

'''
if __name__ == '__main__':
    from torch.autograd import Variable

    net = MLPPolicy(3, 2)

    observation = Variable(torch.randn(2, 3))
    v, action, logprob, mean = net.forward(observation)
    print('v = ', v)
    print(action)
    print('logprob = ', logprob)
    print('mean = ', mean)
    print(net.logstd)
'''
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