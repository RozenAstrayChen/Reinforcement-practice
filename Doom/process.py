import skimage.color
import skimage.transform
from config import *
import numpy as np
import matplotlib.pyplot as plt
import torch
# -*- coding: utf-8 -*-
'''
this is the basic object which is process some chores
'''


class Process:

    def __init__(self):
        pass

    '''
    def save plt
    '''

    def plot_save(self, rewards):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(rewards)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        plt.savefig('./total.jpg')

    '''
    plt reward immediate
    '''

    def plot_durations(self, rewards):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(rewards)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        """
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        """
        plt.pause(0.001)  # pause a bit so that plots are updated

    '''
    show score
    '''

    def show_score(self, scores, num):
        import time
        localtime = time.localtime()
        timeString = time.strftime("%m%d%H", localtime)
        timeString = './' + str(num) + 'score_' + str(timeString) + '.jpg'

        plt.plot(scores)
        plt.xlabel('episodes')
        plt.ylabel('rewads')
        plt.savefig(timeString)
        plt.show()

    '''
    show mean,min,max score
    '''

    def total_score(self, means, mins, maxs):
        name = './' + 'total' + '.jpg'
        plt.plot(means)
        plt.plot(mins)
        plt.plot(maxs)
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        plt.legend(['mean', 'min', 'max'], loc='upper left')
        plt.savefig(name)
        plt.show()

    '''
    Subsampling image and convert to numpy types
    '''

    def preprocess(self, frame):
        #print('show shape',frame.shape)
        # Greyscale frame already done in our vizdoom config
        # x = np.mean(frame,-1)

        # Crop the screen (remove the roof because it contains no information)
        #cropped_frame = frame[30:-10,30:-30]
        # Normalize Pixel Values
        #normalized_frame = frame / 255.0
        # plt.imshow(frame,cmap='gray')
        # plt.show()
        preprocessed_frame = skimage.transform.resize(frame, resolution)
        normalized_frame = preprocessed_frame.astype(np.float32)
        return normalized_frame

    def plot_kernels(self, tensor, layer, num_cols=8, num_rows=6):
        if not tensor.ndim == 4:
            raise Exception("assumes a 4D tensor")
        num_kernels = tensor.shape[1]

        plt.figure(figsize=(num_cols * 2, num_cols * 2))
        for i in range(num_rows, -1, -1):
            for j in range(num_cols, 0, -1):

                #print('i=', i,'j=', j,'\n', 'i*j=',j+(i*4))
                plt.subplot(num_cols, num_rows, j + (i * 4))
                plt.imshow(tensor[0][j], cmap='gray')

            # plt.imshow(tensor[0][i],cmap='gray')
            # plt.show()
        plt.savefig('./' + layer + '.jpg')
        plt.show()

    '''
    see train
    '''

    def plot_grad(self, saliency):
        plt.imshow(np.abs(saliency[0]), cmap='gray')
        # plt.savefig('./'+'saliency'+'.jpg')
        plt.show()

    '''
    save model
    '''

    def save_model(self, name, num, model):
        current_name = './' + str(num) + name + savefile
        torch.save(model, current_name)

    '''
    load model
    '''

    def load_model(self, name, num):
        current_name = './' + str(num) + name + savefile
        print("Loading model from: ", current_name)
        return torch.load(current_name)

    def show_action(self, index):
        if index == 0:
            print('<-- left look')
        elif index == 1:
            print('--> right look')
        else:
            print('^ run')
