# -*- coding: utf-8 -*-
from config import *
import numpy as np
from random import sample
'''
The memory is store experenice, we need collect and sample which is in order to training 
'''


class ReplayMemory:

    def __init__(self, capacity):
        channels = 4
        state_shape = (capacity, channels, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    '''
    Store experenice 
    '''

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    '''
    Sampling experenice
    '''

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]
