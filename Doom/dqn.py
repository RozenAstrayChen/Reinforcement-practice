from random import sample
import itertools as it
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from time import time, sleep
import skimage.color, skimage.transform
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange
import matplotlib.pyplot as plt
'''
my tools
'''
from env import *
from config import *
from memory import *
from net.q_net import Net
from process import Process
from tools.visual import *
print("GPU is ->", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))


class DQN(Process):
    def __init__(self):
        self.game = init_doom(visable=False)
        # find game available action
        n = self.game.get_available_buttons_size()
        #actions = [list(a) for a in it.product([0, 1], repeat=n)]
        actions = np.identity(3, dtype=int).tolist()
        self.action_available = actions
        #self.model = Net(len(actions)).to(device)
        self.model = Net(len(actions)).to(device)
        #loss
        self.criterion = nn.MSELoss().cuda()
        #bp
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         learning_rate)
        self.eps = epsilon
        self.memory = ReplayMemory(replay_memory_size)

    def perform_learning_step(self, load=False, model=0, iterators=1):
        train_mean = []
        train_max = []
        train_min = []
        if load == True:
            self.load_model(model)
        for iterator in range(0, iterators):

            #collect_scores = []

            for epoch in range(epochs):
                print("\nEpoch %d\n-------" % (epoch + 1))
                train_scores = []

                train_episodes_finished = 0
                print("Training...")
                self.game.new_episode()
                # trange show the long process text
                for learning_step in trange(
                        learning_step_per_epoch, leave=False):
                    #while not self.game.is_episode_finished():
                    s1 = self.preprocess(self.game.get_state().screen_buffer)
                    s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
                    action_index = self.choose_action(s1)
                    reward = self.game.make_action(
                        self.action_available[action_index], frame_repeat)

                    isterminal = self.game.is_episode_finished()
                    if not isterminal:
                        s2 = self.preprocess(self.game.get_state()
                                             .screen_buffer)
                        s2 = s2.reshape([1, 1, resolution[0], resolution[1]])
                    else:
                        s2 = None

                    self.memory.add_transition(s1, action_index, s2,
                                               isterminal, reward)

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

            self.save_model(iterator + 1)
        self.total_score(train_mean, train_min, train_max)
        self.game.close()

    '''
    test step
    '''

    def watch_model(self, num):
        self.load_model(num)
        self.game = init_doom(visable=True)
        for _ in range(watch_step_per_epoch):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                state = self.preprocess(self.game.get_state().screen_buffer)
                state = state.reshape([1, 1, resolution[0], resolution[1]])

                action_index = self.choose_action(state, watch_flag=True)
                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                self.game.set_action(self.action_available[action_index])

                for _ in range(frame_repeat):
                    reward = self.game.advance_action()

                #reward = self.game.make_action(self.action_available[action_index])
            sleep(1.0)
            score = self.game.get_total_reward()
            print("Total score: ", score)

        self.game.close()

    def visualization_fliter(self, num):
        self.load_model(num)
        self.game = init_doom(visable=False)
        self.game.new_episode()

        state = self.preprocess(self.game.get_state().screen_buffer)
        state = state.reshape([1, 1, resolution[0], resolution[1]])

        action_index = self.choose_action(state, watch_flag=True)
        self.got_feature(state)
        #self.plot_grad(state[0])

    '''
    save model
    '''

    def save_model(self, num):
        current_name = './' + str(num) + model_savefile
        torch.save(self.model, current_name)

    '''
    load model
    '''

    def load_model(self, num):
        current_name = './' + str(num) + model_savefile
        print("Loading model from: ", current_name)
        self.model = torch.load(current_name)

    '''
    bp using
    process:
        first do forward propagation.
        target_q is actually value, output is predict value
        loss = MSE(output - target_q)
        do optimizer...
    '''

    def propagation(self, state, target_q):
        s1 = torch.from_numpy(state)

        #print('target_q = ',target_q)
        target_q = torch.from_numpy(target_q)
        s1, target_q = Variable(s1), Variable(target_q)
        # change to gpu type
        s1 = s1.cuda(async=True)
        target_q = target_q.cuda(async=True)

        s1 = Parameter(s1.cuda(), requires_grad=True)
        if s1.grad is not None:
            s1.grad.data.zero_()
        '''
        forward propagation
        '''
        output = self.model(s1)
        #print('predict = ',output)
        '''
        get loos value
        '''
        loss = self.criterion(output, target_q).cuda()
        #print('loss = ',loss)
        '''
        do back propagation
        '''
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        '''
        testing
        '''
        #self.plot_grad(s1.grad.data.cpu().numpy()[0])
        return loss

    '''
    convert to tensor data and do forward, 
    and wont do forwad propagation, because is just choice action
    '''

    def get_q(self, state):
        state = torch.from_numpy(state)
        state = Variable(state)
        state = state.to(device)
        #print(state.shape)
        return self.model(state)

    '''
    choose action
    '''

    #(64, 1, 30, 45)
    def choose_action(self, state, watch_flag=False):
        #state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        if self.eps > np.random.uniform() and watch_flag is False:
            action_index = np.random.randint(0, len(self.action_available) - 1)
            '''
            test
            '''
            # using detach is not do forward propagation
            return action_index
        else:
            q = self.get_q(state)
            m, index = torch.max(q, 1)
            action = index.data.cpu().numpy()[0]
            #print('eps == ' ,self.eps,'action ==',action)
            return action

    def got_feature(self, state):
        print('-----------split-----------')
        plt.imshow(state)
        plt.savefig('./' + 'now_state' + '.jpg')
        '''
        conv start
        '''
        state = self.change_cuda(state)
        conv1 = F.relu(self.model.conv1(state))
        state = conv1.data.cpu().numpy()
        self.plot_kernels(state, 'conv1')

        state = self.change_cuda(state)
        conv2 = F.relu(self.model.conv2(state))
        state = conv2.data.cpu().numpy()
        self.plot_kernels(state, 'conv2', num_cols=8, num_rows=16)

    def change_cuda(self, state):
        state = torch.from_numpy(state)
        state = Variable(state)
        state = state.to(device)
        return state

    '''
    learn
    '''

    def learn_from_memory(self):
        # decrease explore rate
        self.exploration_rate()
        if self.memory.size > batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(batch_size)
            #convert numpy type
            target_q = self.get_q(s1).cpu().data.cpu().numpy()
            # get state+1 value
            predict_q = self.get_q(s2).data.cpu().numpy()
            predict_q = np.max(predict_q, axis=1)
            #q_target = r + gamma*(q_next(1)[0].view(batch_size, 1))
            target_q[np.arange(target_q.shape[0]),
                     a] = r + gamma * (1 - isterminal) * predict_q
            self.propagation(s1, target_q)

    '''
    decrease explorate
    '''

    def exploration_rate(self):

        if self.eps > min_eps:
            self.eps *= dec_eps


trainer = DQN()
trainer.perform_learning_step(load=False, iterators=1)
trainer.watch_model(1)
#trainer.visualization_fliter(1)
#plot_kernels(trainer.model.conv1)
