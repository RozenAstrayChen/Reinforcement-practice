import numpy as np

class Rollout(object):
    def __init__(self,):
        self.s = []
        self.a = []
        self.r = []
        #self.n_s = []
    
    def sample(self,):
        s = np.array(self.s)
        a = np.array(self.a)
        r = np.array(self.r)
        '''
        s = np.vstack(self.s)
        a = np.vstack(self.a)
        r = np.vstack(self.r)
        '''
        return s, a, r
    
    def append(self, s, a, r):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)

    def clean(self,):
        self.s = []
        self.a = []
        self.r = []
        