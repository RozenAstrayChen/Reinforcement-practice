import os

import gym
import numpy as np
import tensorboardX
import torch
import torch.nn as nn
from ppo_algor import ppo_update, generate_trajectory, test_result
from torch.optim import Adam
from tqdm import trange

from ppo_net import MLPPolicy

if __name__ == '__main__':
    #env_name = 'BipedalWalker-v2'
    #env_name = 'Pendulum-v0'
    env_name = 'CartPole-v0'
    coeff_entropy = 1e-4
    lr = 2e-4
    mini_batch_size = 64
    horizon = 2048
    nupdates = 5
    nepoch = 4000
    clip_value = 0.2
    policy_path = '{}/lr_{}/coeff_entropy_{}/batch_size_{}/horizon_{}'. \
        format(env_name, lr, coeff_entropy, mini_batch_size, horizon)
    writer = tensorboardX.SummaryWriter(policy_path + '/log/')
    env = gym.make(env_name)
    policy = MLPPolicy(env.observation_space.shape[0],
                       env.action_space.n)
    policy.cuda()
    opt = Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()

    if not os.path.exists(policy_path):
        os.makedirs(policy_path)

    for e in trange(nepoch, leave=False):
        '''
        if e % 100 == 0 and e != 0:
            is_render = True
        else:
            is_render = False
        '''
        
        observations, actions, logprobs, returns, values, rewards =\
            generate_trajectory(env, policy, horizon)

        writer.add_scalar('ppo/mean_reward', np.mean(rewards), global_step=e)
        memory = (observations, actions, logprobs, returns[:-1], values)
        ppo_update(policy, opt, mini_batch_size, memory, nupdates,
                   coeff_entropy=coeff_entropy, clip_value=clip_value)
        print('Episode {}'.format(e))

        if e % 10 == 0 and e != 0:
            rewards = test_result(env, policy, 200)
            print('test reward = ', rewards)
        # save every epoch
        torch.save(policy.state_dict(), policy_path + '/policy.pth')
