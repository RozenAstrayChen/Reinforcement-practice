# -*- coding: utf-8 -*-
from config import *
import numpy as np
from random import sample
from collections import deque  # Ordered collection with ends

'''
The memory is store experenice, we need collect and sample which is in order to training 
'''


class ReplayMemory:

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    '''
    Store experenice 
    '''

    def add_transition(self, experenice):
        self.buffer.append(experenice)
    '''
    Sampling experenice
    '''

    def get_sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]
