import gym
import numpy as np

env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(state_size,action_size)