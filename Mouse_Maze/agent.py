import  numpy as np
import Mouse_Maze.env as Env
import random
import pandas as pd

class Agent:
    def build_table(self,s,a):
        return np.zeros([s,a])
    def __init__(self):
        self.gamma = 0.9
        self.decay =0.9
        self.decay_max = 0.9
        self.decay_min = 0.1
        self.env = Env.env()
        self.table = self.build_table(6,4)

    def random_go(self):
        for i in range(0, 1):
            state = self.env.reset()
            for j in range(0,10):
                action = random.randint(0,3)
                print('-----action = ',action,'-----------')
                state, reward , state_, done = self.env.step(action)
                print('state = ', state)
                print('reward = ', reward)
                print('next_state = ', state_)
                print('done = ', done)
                if done:
                    print('-------------game over--------------')
                    break
    '''
    choose action
    '''
    def choose(self,state):
        # greedy
        if np.random.uniform() < self.decay:
            action =  random.randint(0,4)
        else:
           action = np.argmax(self.table[state])

        return action
    '''
    RL 
    '''
    def learning(self, s, a, r, s_):
        '''
        self.table[s_ + (a)] += self.gamma * (
                r + self.decay * (np.argmax(self.table[s])) - self.table[s_ + (a,)]
        )
        '''
        print(np.argmax(self.table[s]))
        temp = self.table[s][a] + self.gamma *(( r + self.decay * np.argmax(self.table[s])) - self.table[s][a])
        print(temp)

        if self.decay > self.decay_min:
            self.decay = self.decay * 0.99


