import skimage.color, skimage.transform
from config import *
import numpy as np
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
'''
this is the basic object which is process some chores
'''
class Process:
    def __init__(self):
        pass
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
        #plt.imshow(frame,cmap='gray')
        #plt.show()
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

            #plt.imshow(tensor[0][i],cmap='gray')
            #plt.show()
        plt.savefig('./' + layer + '.jpg')
        plt.show()

    '''
    see train
    '''

    def plot_grad(self, saliency):
        plt.imshow(np.abs(saliency[0]), cmap='gray')
        #plt.savefig('./'+'saliency'+'.jpg')
        plt.show()
