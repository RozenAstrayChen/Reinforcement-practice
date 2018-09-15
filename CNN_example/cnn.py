#%%
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt
#%%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # input shape(1, 28, 28)
                nn.Conv2d(
                        in_channels=1, # gray, if image is colorful which channel=3
                        out_channels=16, # number of filter, output is (16,?,?)
                        kernel_size=5, # filter size, output is 32 -5 + 1 = 28
                        stride=1, # filter movement/step
                        padding=2 # 28 + 2*2 = 32,after padding shape become (1,32,32).
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential( # input shape(16,14,14)
                nn.Conv2d(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=5,
                        stride=1,
                        padding=2
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2) # output is (32,7,7)
        )
        self.out = nn.Linear(32*7*7,10) # fully connected layer 10 classes
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
#%%