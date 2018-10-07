from random import sample
import itertools as it
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from time import time, sleep
import skimage.color
import skimage.transform
from torchvision import datasets, transforms
import math
from torch.autograd import Variable
from tqdm import trange
import matplotlib.pyplot as plt
#from torch.distributions import Bernoulli
from torch.distributions import Categorical
import matplotlib.pyplot as plt
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

    def __init__(self, map=map_health):
        self.map = map
        self.game = init_doom(map, visable=False)
        n = self.game.get_available_buttons_size()
        actions = [list(a) for a in it.product([0, 1], repeat=n)]
        #actions = np.identity(3, dtype=int).tolist()
        self.action_available = actions
        #self.model = Net(len(actions)).to(device)
        self.model = Net(len(actions)).to(device)
        # bp
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.eps = epsilon
        # Batch History
        self.eps = np.finfo(np.float32).eps.item()

        self.saved_log_probs = []
        self.rewards = []

    '''
    
    def preprocess(self,frame):
        # Greyscale frame already done in our vizdoom config
        # x = np.mean(frame,-1)
        
        # Crop the screen (remove the roof because it contains no information)
        # [Up: Down, Left: right]
        cropped_frame = frame[80:,:]
        
        # Normalize Pixel Values
        normalized_frame = cropped_frame/255.0
        # Resize
        preprocessed_frame = skimage.transform.resize(normalized_frame, resolution)
        normalized_frame = preprocessed_frame.astype(np.float32)
        return normalized_frame
    '''
    '''
    convert to tensor data and do forward, 
    and wont do forwad propagation, because is just choice action
    '''

    def convert2Tensor(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = Variable(state)
        state = state.to(device)
        # print(state.shape)
        return self.model(state)

    '''
    choose action
    '''

    def choose_action(self, state):

        probs = self.convert2Tensor(state)
        # print(probs)
        m = Categorical(probs=probs)
        action = m.sample()
        '''
        action = torch.multinomial(m.probs, 1, True)
        '''
        self.saved_log_probs.append(m.log_prob(action))

        # print(action)

        return action.item()

    '''
    REINFORCE algorithim
    '''

    def update_poilicy(self, ):
        #print('update policy start')
        R = 0
        rewards = []
        policy_loss = []

        for r in self.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.tensor(rewards)
        # if basic will shot immediate, reward - reward.mean is zero
        # if rewards.shape[0] != 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            log_prob = log_prob.to(device)
            reward = reward.to(device)
            policy_loss.append(-log_prob * reward)

        # else:
        #    rewards = math.log(rewards)
        #   rewards = torch.FloatTensor([rewards])

    # loss = (torch.sum(torch.mul(self.policy_history, Variable(rewards)).mul(-1), -1))
        policy_loss = torch.cat(policy_loss).sum()
        '''
        print('policy = ',self.saved_log_probs)
        print('rewards = ',rewards)
        print('loss = ',policy_loss)
        '''

        # Update netowrk weight
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Save and intialize episode history counters
        del self.rewards[:]
        del self.saved_log_probs[:]
    '''
    train step
    '''

    def train_model(self, load=False, num=0, iterators=1):
        if load == True:
            self.model = self.load_model(policy, num)
        train_episodes_finished = 0
        reward_collect = []
        for iterator in range(0, iterators):

            for epoch in range(learning_step_per_epoch):
                self.game.new_episode()
                train_scores = []
                for learning_step in range(1000):
                    s1 = self.preprocess(self.game.get_state().screen_buffer)

                    s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
                    action_index = self.choose_action(s1)

                    reward = self.game.make_action(
                        self.action_available[action_index], frame_repeat)

                    self.rewards.append(reward)

                    if self.game.is_episode_finished():
                        train_episodes_finished += 1
                        train_scores.append(self.game.get_total_reward())

                        break
                # update data
                self.update_poilicy()
                if (train_episodes_finished % 50 == 0):
                    print("%d training episodes played." %
                          train_episodes_finished)
                    reward_collect.append(train_scores)
                    train_scores = np.array(train_scores)
                    print("Results: mean: %.1f +/- %.1f," %
                          (train_scores.mean(), train_scores.std()))
                    self.plot_durations(reward_collect)
            self.save_model(iterator + 1, self.model)
        self.plot_save(policy, reward_collect)
        self.game.close()

    '''
    test step
    '''

    def watch_model(self, num, delay=False):
        import time
        self.model = self.load_model(policy_gradient, num)
        self.game = init_doom(scenarios=self.map, visable=True)
        for _ in range(watch_step_per_epoch):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                state = self.preprocess(self.game.get_state().screen_buffer)
                state = state.reshape([1, 1, resolution[0], resolution[1]])
                action_index = self.choose_action(state)
                if delay is True:
                    self.show_action(action_index)
                    sleep(0.5)
                # print(action_index)
                # Instead of make_action(a, frame_repeat) in order to make the
                # animation smooth
                self.game.set_action(self.action_available[action_index])
                reward = self.game.advance_action()

                #reward = self.game.make_action(self.action_available[action_index])
            sleep(1.0)
            score = self.game.get_total_reward()
            print("Total score: ", score)

        self.game.close()


'''
policy = Policy()
#policy.train_model(load=False, num=2, iterators=5)

policy.watch_model(5,delay=False)
print('over!')
'''
