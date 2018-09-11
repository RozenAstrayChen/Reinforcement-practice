#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
#%%
# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9 # greedy policy
GAMMA = 0.9 # alpha
TARGET_REPLACE_ITER = 100 # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
# using GPU
print("GPU is ->",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))
#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0,0.1)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
#%%
class DQN:
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        #self.eval_net.cuda()
        #self.target_net.cuda()
        
        self.learn_step_counter = 0 # for target update
        self.memory_counter = 0 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2+2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
    def choose_action(self,x):
        x = torch.unsqueeze(torch.FloatTensor(x),0).to(device) # input data to tensor
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE) 
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action
    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s, [a,r],s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index,:] = transition
        self.memory_counter += 1
    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter +=1
        
        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        # need using cuda
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).cuda() # 0..3
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).cuda() #4
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).cuda() #5
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).cuda() #6..9
        '''
        print('b_s:',b_s)
        print('b_a:',b_a)
        print('b_r:',b_r)
        print('b_s_:',b_s_)
        '''
         # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        '''
        print('q_eval:',q_eval)
        print('q_next:',q_next)
        print(q_next.max(1)[0].view(BATCH_SIZE, 1))
        print('q_target:',q_target)
        print('loss:',loss)
        '''
        print('bs:',b_s)
        print('eval_net(b_s):',self.eval_net(b_s))
        print('eval_net(b_s).gather(1, b_a):',self.eval_net(b_s).gather(1, b_a))
        
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
#%%
dqn = DQN()


print('\nCollecting experience...')
reward_collection = []
for i_episode in range(1000):
    s = env.reset()
    ep_r = 0
    while True:
        #env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.unwrapped.x_threshold - abs(x)) / env.unwrapped.x_threshold - 0.8
        r2 = (env.unwrapped.theta_threshold_radians - abs(theta)) / env.unwrapped.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
                reward_collection.append(ep_r)

        if done:
            break
        s = s_
#%%
plt.plot(reward_collection)
plt.ylabel('total rewads')
plt.show()
#%%
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
#%%
