import os

import gym
import numpy as np
import tensorboardX
import torch
import torch.nn as nn
from ppo_v2 import ppo_update, generate_trajectory
from torch.optim import Adam

from net import MLPPolicy

if __name__ == '__main__':
    env_name = 'BipedalWalker-v2'
    #env_name = 'Pendulum-v0'
    #env_name = 'MountainCar-v0'
    coeff_entropy = 1e-4
    lr = 2e-4
    mini_batch_size = 64
    horizon = 2048
    nupdates = 5
    nepoch = 40000
    clip_value = 0.2
    policy_path = '{}/lr_{}/coeff_entropy_{}/batch_size_{}/horizon_{}'. \
        format(env_name, lr, coeff_entropy, mini_batch_size, horizon)
    writer = tensorboardX.SummaryWriter(policy_path + '/log/')
    env = gym.make(env_name)
    policy = MLPPolicy(env.observation_space.shape[0],
                       env.action_space.shape[0])
    policy.cuda()
    opt = Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()

    if not os.path.exists(policy_path):
        os.makedirs(policy_path)

    for e in range(nepoch):
        if e % 100 == 0 and e != 0:
            is_render = True
        else:
            is_render = False
        observations, actions, logprobs, returns, values, rewards =\
            generate_trajectory(env, policy, horizon, is_render=is_render)

        writer.add_scalar('ppo/mean_reward', np.mean(rewards), global_step=e)
        memory = (observations, actions, logprobs, returns[:-1], values)
        value_loss, policy_loss, dist_entropy = ppo_update(
            policy,
            opt,
            mini_batch_size,
            memory,
            nupdates,
            coeff_entropy=coeff_entropy,
            clip_value=clip_value)
        if e % 20 == 0:
            print(
                'Episode {}, reward is {} \nvalue_loss : {} \t\t policy_loss : {} \t\t entropy : {}\n'.
                format(e, rewards.sum(), value_loss, policy_loss, dist_entropy))
            # save every epoch
            torch.save(policy.state_dict(), policy_path + '/policy.pth')
