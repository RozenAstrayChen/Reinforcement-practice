from random import sample
import itertools as it
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from time import time, sleep
import skimage.color, skimage.transform
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange
import matplotlib.pyplot as plt
#from torch.distributions import Bernoulli
from torch.distributions import Categorical
'''
my tools
'''
from env import *
from config import *
from memory import *
from net.pg_net import *
from process import Process
from tools.visual import *

print("GPU is ->", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

class Policy(Process):
    def __init__(self):
        self.game = init_doom(visable=False)
        n = self.game.get_available_buttons_size()
        #actions = [list(a) for a in it.product([0, 1], repeat=n)]
        actions = np.identity(3,dtype=int).tolist()
        self.action_available = actions
        #self.model = Net(len(actions)).to(device)
        self.model = Net(len(actions))
        #loss
        self.criterion = nn.MSELoss().cuda()
        #bp
        self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                         learning_rate)
        self.eps = epsilon
        # Batch History
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.steps = 0
    
    '''
    convert to tensor data and do forward, 
    and wont do forwad propagation, because is just choice action
    '''
    def convert2Tensor(self, state):
        state = torch.from_numpy(state)
        state = Variable(state)
        #state = state.to(device)
        #print(state.shape)
        return self.model(state)
    '''
    choose action
    '''
    def choose_action(self,state):
        probs = self.convert2Tensor(state)
        m = Categorical(probs=probs)
        
        
        action = torch.multinomial(m.probs, 1, True)
        #action = m.sample()
        
        action = action.data.numpy().astype(int)[0]
        #print(action[0])
        
        return action[0]
        
    def update_poilicy(self,):
        #print(self.steps)
        reward_pool = np.array(self.reward_pool)
        #print('shape',reward_pool.shape)
        # discount_reward
        running_add =0
        for i in reversed(range(self.steps)):
            if self.reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * gamma + self.reward_pool[i]
                self.reward_pool[i] = running_add
        # Normalize reward
        reward_mean = np.mean(self.reward_pool)
        reward_std = np.std(self.reward_pool)
        for i in range(self.steps):
            self.reward_pool[i] = (self.reward_pool[i] - reward_mean) / reward_std
        # gradient desent
        self.optimizer.zero_grad()
        for i in range(self.steps):
            state = self.state_pool[i]
            action = Variable(torch.FloatTensor([self.action_pool[i]]))
            rewad = self.reward_pool[i]
            
            probs = self.convert2Tensor(state)
            m = Categorical(probs)
            loss = -m.log_prob(action) * rewad # Negtive score function x reward
            loss.backward()
        self.optimizer.step()
        
        self.reward_pool = []
        self.state_pool = []
        self.action_pool = []
        self.steps = 0
        
    '''
    train step
    '''
    def train_model(self):
        train_episodes_finished=0
        
        for epoch in range(epochs):
            self.game.new_episode()
            train_scores = []
            for learning_step in trange(learning_step_per_epoch, leave=False):
                s1 = self.preprocess(self.game.get_state().screen_buffer)
                s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
                action_index = self.choose_action(s1)
                reward = self.game.make_action(
                        self.action_available[action_index], frame_repeat)
                self.trainistion(s1,action_index,reward)
                
                isterminal = self.game.is_episode_finished()
                if self.game.is_episode_finished():
                    train_episodes_finished+=1
                    score = self.game.get_total_reward()
                    train_scores.append(score)
                    self.update_poilicy()
                    self.game.new_episode()
                self.steps += 1
            print("%d training episodes played." % train_episodes_finished)
            train_scores = np.array(train_scores)
            print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
    '''
    test step
    '''
    def watch_model(self):
        self.game.new_episode()
        #while not self.game.is_episode_finished():
        state = self.preprocess(self.game.get_state().screen_buffer)
        state = state.reshape([1, 1, resolution[0], resolution[1]])
        
        action_index = self.choose_action(state)
            #print(action_index)
        self.game.close()
        return
    def trainistion(self,state,action,reward):
        self.state_pool.append(state)
        self.action_pool.append(action)
        self.reward_pool.append(reward)
            
policy = Policy()
policy.train_model()
print('over!')