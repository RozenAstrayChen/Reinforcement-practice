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
from torch.distributions import Categorical

'''
my tools
'''
from env import *
from config import *
from memory import *
from net import *
from process import Process
from tools.visual import *

print("GPU is ->", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

class Policy(Process):
    def __init__(self):
        self.game = init_doom(visable=False)
        n = self.game.get_available_buttons_size()
        actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.action_available = actions
        #self.model = Net(len(actions)).to(device)
        self.model = Net(len(actions))
        #loss
        self.criterion = nn.MSELoss().cuda()
        #bp
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         learning_rate)
        self.eps = epsilon
        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
    '''
    convert to tensor data and do forward, 
    and wont do forwad propagation, because is just choice action
    '''
    def convert2Tensor(self, state):
        state = torch.from_numpy(state)
        state = Variable(state)
        state = state
        #print(state.shape)
        return self.model(state)
    '''
    choose action
    '''
    def choose_action(self,state):
        # base on probabilities in state,difference in value basic,
        state = self.convert2Tensor(state)
        c = Categorical(state)
        action = c.sample()
        #print(action)
        
        # Add log probability of our choosen action to our history
        if self.policy_history.dim() != 0:
            self.policy_history = torch.cat([self.policy_history,c.log_prob(action)])
        else:
            self.policy_history = (c.log_prob(action))
        
        return action
    def update_poilicy(self):
        R = 0
        rewards = []
        # Discount futrue rewards back using gamma
        for r in self.reward_episode[::-1]:
            R = r + gamma * R
            rewards.insert(0,R)
        rewards = torch.FloatTensor(rewards)
        #print('rewards =',rewards)
        #print('mean =',rewards.mean())
        #print('rewards.std()',rewards.std())
        #print('np.finifo',np.finfo(np.float32).eps)
        # Scale rewards
        rewards = (rewards - rewards.mean()) / (rewards.std())
        #rewards = (rewards - rewards.mean()) / (rewards.std())
        # Calculate loss
        loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #Save and intialize episode history counters
        self.loss_history.append(loss.data[0])
        self.reward_history.append(np.sum(policy.reward_episode))
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode= []
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
                s1 = s1.reshape([1, 3, resolution[0], resolution[1]])
                action_index = self.choose_action(s1)
                reward = self.game.make_action(
                        self.action_available[action_index], frame_repeat)
                self.reward_episode.append(reward)
                isterminal = self.game.is_episode_finished()
                if self.game.is_episode_finished():
                    train_episodes_finished+=1
                    score = self.game.get_total_reward()
                    train_scores.append(score)
                    self.update_poilicy()
                    self.game.new_episode()
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
        state = state.reshape([1, 3, resolution[0], resolution[1]])
        
        action_index = self.choose_action(state)
            #print(action_index)
        self.game.close()
        return
            
policy = Policy()
policy.train_model()
print('over!')