# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#%%
#%%
class Net(nn.Module):
    def __init__(self,N_STATES,N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0,0.1)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
#%%
#%%
