import numpy as np
import random


class Rollout(object):
    def __init__(self,batch=32):
        self.flush()
        self.batch = batch
    
    def sample(self):
        idx = np.arange(0 , len(self.r))
        np.random.shuffle(idx)
        idx = idx[:self.batch]
        s = [self.s[i] for i in idx]
        a = [self.a[i] for i in idx]
        #r = [self.r[i] for i in idx]
        #n_s = [self.n_s[i] for i in idx]
        #done = [[self.done[i] for i in idx]]
        adv = [[self.adv[i] for i in idx]]

        s = np.array(s)
        a = np.array(a)
        adv = np.array(adv)


        return s, a, adv

    def append(self, s, a, r, n_s, d):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.n_s.append(n_s)
        self.done.append(d)

    def flush(self,):
        self.s = []
        self.a = []
        self.r = []
        self.n_s = []
        self.done = []
        self.adv = []
    
    def adv_replace(self, adv):
        self.adv = adv
        