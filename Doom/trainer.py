from env import *
from config import *
from memory import *
from net import *
import itertools as it
import torch
import torch.nn.functional as F
import torch.nn as nn
import skimage.color, skimage.transform
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange

print("GPU is ->",torch.cuda.is_available())                                                                                                                                         
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                                                                                                
print(torch.cuda.get_device_name(0))

class Trainer:
    def __init__(self):
        self.game = init_doom(visable=True)
        # find game available action
        n = self.game.get_available_buttons_size()
        actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.action_available = actions
        self.model = Net(len(actions)).to(device)
        #loss
        criterion = nn.MSELoss()
        #bp
        self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate)
        self.eps =  epsilon
        self.memory = ReplayMemory(replay_memory_size)
        

        
    def perform_learning_step(self):
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            print("Training...")
            self.game.new_episode()
            while not self.game.is_episode_finished():
                s1 = self.preprocess(self.game.get_state().screen_buffer)
                s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
                
                action_index = self.choose_action(s1)
                reward = self.game.make_action(self.action_available[action_index], frame_repeat)
                
                isterminal = self.game.is_episode_finished()
                s2 = self.preprocess(self.game.get_state().screen_buffer) if not isterminal else None
                
                self.memory.add_transition(s1, action_index, s2, isterminal, reward)
                
            if self.game.is_episode_finished():
                    score = game.get_total_reward()
                    print('score = ',score)
        self.game.close()
    '''
    Subsampling image and convert to numpy types
    '''
    def preprocess(self,img):
        img = skimage.transform.resize(img, resolution)
        img = img.astype(np.float32)
        return img
    '''
    bp using
    '''
    def back_propagation(actually,predict):
        pass
    '''
    choose action
    '''
    #(64, 1, 30, 45)
    def choose_action(self,state):
        state = torch.unsqueeze(torch.FloatTensor(state),0).to(device)
        #if self.eps > np.random.uniform():
        action_index = np.random.randint(0,len(self.action_available)-1)
        return action_index
        #else:
        #    action_value = self.model(state)
        #    action = torch.max(action_value, 1)[1].data.cpu().numpy()
        #    return action
        
            
    '''
    learn
    '''
    def learn(self):
        self.exploration_rate()
        
    '''
    decrease explorate
    '''
    def exploration_rate(self):
        self.eps = self.eps * dec_eps
        if self.eps < min_epsilon:
            self.eps = min_epsilon
        
        

trainer = Trainer()
trainer.perform_learning_step()
