#%%
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision # data moudle
import matplotlib.pyplot as plt
from cnn import CNN

torch.manual_seed(1) # reproducible
#%%
# hyperparameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
#%%
train_data = torchvision.datasets.MNIST(root='./mnist/',train=True,transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
#%%
test_data = torchvision.datasets.MNIST(root='./mnist/',train=False)
#train
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]
#%%
cnn = CNN()
print(cnn)
#%%
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   #  batch data, normalize x when iterate train_loader
        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
#%%
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
#%%