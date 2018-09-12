#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net import Net
#%%
# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9 # greedy policy
GAMMA = 0.9 # alpha
TARGET_REPLACE_ITER = 100 # target update frequency
MEMORY_CAPACITY = 2000

# using GPU
print("GPU is ->",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

class DDQN:
    def __init__(self,N_STATES,N_ACTIONS,ENV_A_SHAPE):
        self.eval_net, self.target_net = Net(N_STATES,N_ACTIONS).to(device), Net(N_STATES,N_ACTIONS).to(device)
        #self.eval_net.cuda()
        #self.target_net.cuda()
        
        self.learn_step_counter = 0 # for target update
        self.memory_counter = 0 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2+2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.ENV_A_SHAPE = ENV_A_SHAPE
    def choose_action(self,x):
        x = torch.unsqueeze(torch.FloatTensor(x),0).to(device) # input data to tensor
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()
            action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE) 
        else:
            action = np.random.randint(0, self.N_ACTIONS)
            action = action if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
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
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES]).cuda() # 0..3
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int)).cuda() #4
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2]).cuda() #5
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:]).cuda() #6..9
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
        '''
        print('bs:',b_s)
        print('eval_net(b_s):',self.eval_net(b_s))
        print('eval_net(b_s).gather(1, b_a):',self.eval_net(b_s).gather(1, b_a))
        '''
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

#%%
