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
from collections import namedtuple
'''
my tools
'''
from env import *
from config import *
from memory import *
from net.ac_net import ACNet
from process import *
from tools.visual import *
from policy import *
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class AC(Policy):

    def __init__(self, map=map_health):
        super(AC, self).__init__(map)
        self.map = map
        self.model = ACNet(len(self.action_available)).to(self.device)
        self.saved_actions = []
        self.rewards = []
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)

    def convert2Tensor(self, state):
        state = torch.from_numpy(state)
        state = Variable(state)
        state = state.to(self.device)
        # print(state.shape)
        return self.model(state)

    '''
    Overwirte choose action
    '''

    def choose_action(self, state):
        state = state.reshape([1, 3, resolution[0], resolution[1]])
        probs, state_value = self.convert2Tensor(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action

    '''
    Overwrite update policy
    '''

    def update_policy(self, ):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        # REINFORCE
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.item()

            log_prob = log_prob.to(self.device)
            reward = reward.to(self.device)

            policy_losses.append(-log_prob * reward)
            value_losses.append(
                F.smooth_l1_loss(value[0],
                                 torch.tensor([r]).to(self.device)))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(
            value_losses).sum()
        loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]

    '''
    Overwrite train model
    '''

    def train_model(self, load=False, num=0, iterators=1):
        if load is True:
            self.model = self.load_model(actor_cirtic, num)
        train_episodes_finished = 0
        rewards_collect = []
        for iterator in range(0, iterators):
            for epoch in range(total_episode):
                self.game.new_episode()
                train_scores = []
                for _ in range(learning_step_per_epoch):
                    s1 = self.preprocess(self.game.get_state().screen_buffer)
                    s1 = self.frames_reshape(s1)

                    action_index = self.choose_action(s1)
                    self.game.set_action(
                        self.action_available[action_index])
                    self.game.advance_action(frame_skip)
                    reward = self.game.get_last_reward()

                    self.rewards.append(reward)

                    if self.game.is_episode_finished():
                        train_episodes_finished += 1
                        train_scores.append(self.game.get_total_reward())

                        break
                self.update_policy()
                if (train_episodes_finished % 20 == 0):
                    print("%d training episodes played." %
                          train_episodes_finished)
                    rewards_collect.append(train_scores)
                    train_scores = np.array(train_scores)
                    print("Results: mean: %.1f +/- %.1f," %
                          (train_scores.mean(), train_scores.std()))
                    self.plot_durations(rewards_collect)
            self.save_model(actor_cirtic, iterator + 1, self.model)
        self.plot_save(rewards_collect)
        self.game.close()

    def watch_model(self, num, delay=False):
        import time
        self.model = self.load_model(actor_cirtic, num)
        self.game = init_doom(scenarios=self.map, visable=True)
        for _ in range(watch_step_per_epoch):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                state = self.preprocess(self.game.get_state().screen_buffer)
                state = self.frames_reshape(state)
                action_index = self.choose_action(state)
                self.game.set_action(self.action_available[action_index])
                self.game.advance_action(frame_skip)

                #reward = self.game.make_action(self.action_available[action_index])
            sleep(1.0)
            score = self.game.get_total_reward()
            print("Total score: ", score)

        self.game.close()

'''
ac = AC()
#ac.train_model(load=True, num=1, iterators=5)
ac.watch_model(5)
'''
