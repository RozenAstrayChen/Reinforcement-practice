# -*- coding: utf-8 -*-
import itertools as it
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_kernels(tensor, num_cols=1):
    tensor = tensor.weight.data.cpu().numpy()
    print(tensor.shape)
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    
    if not tensor.shape[-1]==8:
        raise Exception("last dim needs to be 8 to plot")
    
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        print(i)
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    