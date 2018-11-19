import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
'''
calculate discount rewards
'''


def calculate_returns(rewards, dones, last_value, gamma=0.99):
    # create nparray
    returns = np.zeros(rewards.shape[0] + 1)
    returns[-1] = last_value
    dones = 1 - dones
    for i in reversed(range(rewards.shape[0])):
        returns[i] = gamma * returns[i + 1] * dones[i] + rewards[i]

    return returns


'''
using old pi to sample data
'''


def ppo_update(policy,
               optimizer,
               batch_size,
               memory,
               nupdates,
               coeff_entropy=0.02,
               clip_value=0.2,
               writer=None):
    obs, actions, logprobs, returns, values = memory
    print('ret shape', returns.shape)
    print('value shape', values.shape)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / advantages.std()
    print(advantages.shape)
    print(advantages)
    for update in range(nupdates):
        sampler = BatchSampler(
            SubsetRandomSampler(list(range(advantages.shape[0]))),
            batch_size=batch_size,
            drop_last=False)

        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obs[index])).float().cuda()
            sampled_actions = Variable(torch.from_numpy(
                actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(
                logprobs[index])).float().cuda()
            sampled_returns = Variable(torch.from_numpy(
                returns[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(
                advantages[index])).float().cuda()

            new_value, new_logprob, dist_entropy = policy.evaluate_actions(
                sampled_obs, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value,
                                     1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_returns = sampled_returns.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_returns)

            loss = policy_loss + value_loss - coeff_entropy * dist_entropy
            #print('loss = ', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''
            tensorboardX
            '''
            if writer is not None:
                writer.add_scalar('ppo/value_loss', value_loss.data[0])
                writer.add_scalar('ppo/policy_loss', policy_loss.data[0])
                writer.add_scalar('ppo/entropy', dist_entropy.data[0])
    return value_loss.data[0], policy_loss.data[0], dist_entropy.data[0]


def generate_trajectory(env,
                        policy,
                        max_step,
                        obs_fn=None,
                        progress=False,
                        is_render=False):
    """generate a batch of examples using policy"""
    nstep = 0
    obs = env.reset()
    done = False
    observations, rewards, actions, logprobs, dones, values = [], [], [], [], [], []
    while not (nstep == max_step):
        if done:
            obs = env.reset()
       

        obs = Variable(torch.from_numpy(obs[np.newaxis])).float().cuda()
        value, action, logprob = policy.action(obs)
        value, action, logprob = value.data.cpu().numpy()[
            0], action.data.cpu().numpy()[0], logprob.data.cpu().numpy()[0]
        
        next_obs, reward, done, _ = env.step(action[0])
        
        observations.append(obs.data.cpu().numpy()[0])
        rewards.append(reward)
        logprobs.append(logprob)
        dones.append(done)
        values.append(value[0])
        actions.append(action)

        obs = next_obs
        nstep += 1
        
        if progress:
            print('\r{}/{}'.format(nstep, r_episode), flush=True, end='')
        
    if done:
        last_value = 0.0
    else:
        if obs_fn is not None:
            obs = obs_fn(obs)
        obs = Variable(torch.from_numpy(obs[np.newaxis])).float().cuda()
        value, action, logprob = policy.action(obs)
        last_value = value.data[0][0]
    observations = np.asarray(observations)
    rewards = np.asarray(rewards)
    logprobs = np.asarray(logprobs)
    dones = np.asarray(dones)
    values = np.asarray(values)
    actions = np.asarray(actions)
    returns = calculate_returns(rewards, dones, last_value)
    print('obs', observations.shape)
    print('ret', returns.shape)
    print('value', values.shape)
    return observations, actions, logprobs, returns, values, rewards


def test_result(env,
                policy,
                max_step):
    nstep = 0
    obs = env.reset()
    collect_rwards = 0
    while not (nstep == max_step):
        env.render()

        obs = Variable(torch.from_numpy(obs[np.newaxis])).float().cuda()

        value, action, logprob = policy.action(obs)
        action = action.data.cpu().numpy()[0]
        next_obs, reward, done, _ = env.step(action[0])
        collect_rwards += reward
        obs = next_obs
        nstep += 1

    return collect_rwards


