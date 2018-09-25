from env import *
from config import *
from memory import *
from net import *
from random import sample
import itertools as it
import torch
import torch.nn.functional as F
import torch.nn as nn
from time import time, sleep
import skimage.color, skimage.transform
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange
import matplotlib.pyplot as plt

print("GPU is ->", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))


class Trainer:
    def __init__(self):
        self.game = init_doom(visable=False)
        # find game available action
        n = self.game.get_available_buttons_size()
        actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.action_available = actions
        #self.model = Net(len(actions)).to(device)
        self.model = Net(len(actions))
        #loss
        self.criterion = nn.MSELoss()
        #bp
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         learning_rate)
        self.eps = epsilon
        self.memory = ReplayMemory(replay_memory_size)

    def perform_learning_step(self,load = False,iterators=1):
        train_mean = []
        train_max =[]
        train_min = []
        for iterator in range(2,iterators):
            
            if load == True and iterator != 0:
                self.load_model(iterator)
            #collect_scores = []
            
            for epoch in range(epochs):
                print("\nEpoch %d\n-------" % (epoch + 1))
                train_scores = []
                
                train_episodes_finished = 0
                print("Training...")
                self.game.new_episode()
                # trange show the long process text
                for learning_step in trange(learning_step_per_epoch, leave=False):
                    #while not self.game.is_episode_finished():
                    s1 = self.preprocess(self.game.get_state().screen_buffer)
                    s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
    
                    action_index = self.choose_action(s1)
                    reward = self.game.make_action(
                        self.action_available[action_index], frame_repeat)
    
                    isterminal = self.game.is_episode_finished()
                    s2 = self.preprocess(
                        self.game.get_state()
                        .screen_buffer) if not isterminal else None
    
                    self.memory.add_transition(s1, action_index, s2, isterminal,
                                               reward)
    
                    self.learn_from_memory()
    
                    if self.game.is_episode_finished():
                        score = self.game.get_total_reward()
                        train_scores.append(score)
                        #collect_scores.append(score)
                        train_episodes_finished += 1
                        # next start
                        self.game.new_episode()
                print("%d training episodes played." % train_episodes_finished)
                train_scores = np.array(train_scores)
                print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                      "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
                train_mean.append(train_scores.mean())
                train_min.append(train_scores.min())
                train_max.append(train_scores.max())
            '''
            loop over
            '''
            #self.show_score(collect_scores,iterator)
            
            self.save_model(iterator+1)
        self.total_score(train_mean,train_min,train_max)
        self.game.close()

    '''
    test step
    '''

    def watch_model(self,num):
        self.load_model(num)
        self.game = init_doom(visable=True)
        for _ in range(watch_step_per_epoch):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                state = self.preprocess(self.game.get_state().screen_buffer)
                state = state.reshape([1, 1, resolution[0], resolution[1]])
                action_index = self.choose_action(state,watch_flag=True)
                
                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                self.game.set_action(self.action_available[action_index])
                
                for _ in range(frame_repeat):
                    reward = self.game.advance_action()
                
                #reward = self.game.make_action(self.action_available[action_index])
            sleep(1.0)
            score = self.game.get_total_reward()
            print("Total score: ", score)

        self.game.close()

    '''
    save model
    '''

    def save_model(self,num):
        current_name = './' + str(num) + model_savefile
        torch.save(self.model, current_name)

    '''
    load model
    '''

    def load_model(self,num):
        current_name = './' + str(num) + model_savefile
        print("Loading model from: ", current_name)
        self.model = torch.load(current_name)

    '''
    show score
    '''

    def show_score(self, scores,num):
        import time
        localtime = time.localtime()
        timeString = time.strftime("%m%d%H", localtime)
        timeString = './' + str(num) +'score_' + str(timeString) + '.jpg'

        plt.plot(scores)
        plt.xlabel('episodes')
        plt.ylabel('rewads')
        plt.savefig(timeString)
        plt.show()
    
    '''
    show mean,min,max score
    '''
    def total_score(self,means,mins,maxs):
        name = './'  + 'total' + '.jpg'
        plt.plot(means)
        plt.plot(mins)
        plt.plot(maxs)
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        plt.legend(['mean', 'min','max'], loc='upper left')
        plt.savefig(name)
        plt.show()

    '''
    Subsampling image and convert to numpy types
    '''

    def preprocess(self, img):
        img = skimage.transform.resize(img, resolution)
        img = img.astype(np.float32)
        return img

    '''
    bp using
    '''

    def back_propagation(self, state, target_q):
        s1 = torch.from_numpy(state)
        #print('target_q = ',target_q)
        target_q = torch.from_numpy(target_q)
        s1, target_q = Variable(s1), Variable(target_q)
        output = self.model(s1)
        #print('predict = ',output)
        loss = self.criterion(output, target_q)
        #print('loss = ',loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    '''
    convert to tensor data
    '''

    def tensor_type(self, state):
        state = torch.from_numpy(state)
        state = Variable(state)
        #print(state.shape)
        return self.model(state)

    '''
    choose action
    '''

    #(64, 1, 30, 45)
    def choose_action(self, state,watch_flag=False):
        #state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        if self.eps > np.random.uniform() and watch_flag is False:
            action_index = np.random.randint(0, len(self.action_available) - 1)
            return action_index
        else:

            q = self.tensor_type(state)
            m, index = torch.max(q, 1)
            action = index.data.numpy()[0]
            #print('eps == ' ,self.eps,'action ==',action)
            return action

    '''
    learn
    '''

    def learn_from_memory(self):
        # decrease explore rate
        self.exploration_rate()
        if self.memory.size > batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(batch_size)
            #convert numpy type
            target_q = self.tensor_type(s1).data.numpy()
            #q_eval = self.model(s1).gather(1, a)
            q = self.tensor_type(s2).data.numpy()
            #q_next = self.model(s2).detach() # won't do BP
            q2 = np.max(q, axis=1)
            #q_target = r + gamma*(q_next(1)[0].view(batch_size, 1))
            target_q[np.arange(target_q.shape[0]),
                     a] = r + gamma * (1 - isterminal) * q2
            self.back_propagation(s1, target_q)

    '''
    decrease explorate
    '''

    def exploration_rate(self):

        if self.eps > min_eps:
            self.eps *= dec_eps


trainer = Trainer()
#trainer.perform_learning_step(load=True,iterators=10)
trainer.watch_model(10)