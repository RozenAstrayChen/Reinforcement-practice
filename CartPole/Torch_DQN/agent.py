# -*- coding: utf-8 -*-
from double_dqn import DDQN
from dqn import DQN
import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
#env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v1')
#env = gym.make('Pendulum-v0')
N_ACTIONS = env.action_space.n
#N_ACTIONS = 11
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
#%%


MEMORY_CAPACITY = 2000



def dqn_work(dqn):
    print('\nCollecting experience...')
    rewards = []
    for i_episode in range(400):
        s = env.reset()
        ep_r = 0
        while True:
            #env.render()
            a = dqn.choose_action(s)
    
            # take action
            s_, r, done, info = env.step(a)
            '''
            # modify the reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.unwrapped.x_threshold - abs(x)) / env.unwrapped.x_threshold - 0.8
            r2 = (env.unwrapped.theta_threshold_radians - abs(theta)) / env.unwrapped.theta_threshold_radians - 0.5
            r = r1 + r2
            '''
            '''
            #MountainerCar
            r = abs(s_[0] - (-0.5))
            '''
            dqn.store_transition(s, a, r, s_)
    
            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))
                    
    
            if done:
                rewards.append(ep_r)
                break
            s = s_
    return dqn, rewards
#%%
def show_rewards(d_rewards,dd_rewards):
    import time
    localtime = time.localtime()
    timeString = time.strftime("%m%d%H", localtime)
    timeString = './' + str(timeString) + '.jpg'
    
    plt.plot(d_rewards)
    plt.plot(dd_rewards)
    plt.xlabel('episodes')
    plt.ylabel('total rewads')
    plt.legend(['dqn', 'ddqn'], loc='upper left')
    plt.savefig(timeString)
    plt.show()
#%%
def test(dqn):
    print('\nTesting mode')
    for i_episode in range(10):
        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            a = dqn.choose_action(s)
            # take action
            s_, r ,done, info = env.step(a)
            
            if done:
                break
            s = s_
    env.close()
            
#%%
ddqn = DDQN(N_STATES,N_ACTIONS,ENV_A_SHAPE)
dqn = DQN(N_STATES,N_ACTIONS,ENV_A_SHAPE)

dqn,d_rewards = dqn_work(dqn)
ddqn,dd_rewards = dqn_work(ddqn)
show_rewards(d_rewards,dd_rewards)
#%%
test(dqn)
test(ddqn)
#%%