import tensorflow.keras as ks
import collections
import numpy as np
import random
from tensorflow.keras.models import load_model,save_model


class DQNAgent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.transition_max = 2000

        self.memory = collections.deque(maxlen=self.transition_max)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learn_rate = 0.001
        self.model = self.build_mode()
        self.actions = []

    def build_mode(self):
        model = ks.Sequential()
        model.add(ks.layers.Dense(
            24, input_dim=self.num_states, activation='relu'
        ))
        model.add(ks.layers.Dense(
            24, activation='relu'
        ))
        model.add(ks.layers.Dense(
            self.num_actions, activation='relu'
        ))
        model.compile(loss='mse',
                      optimizer=ks.optimizers.Adam(lr=self.learn_rate))

        return model

    def transition(self, state, action, reward, state_, done):
        self.memory.append((state, action, reward, state_, done))
        if len(self.memory) > self.transition_max:
            self.memory.pop(0)

    def choose(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        act_values = self.model.predict(state)
        action =  np.argmax(act_values[0])
        self.actions.append(action)
        return action
    def replay(self, batch_size):
        
        minibatch = random.sample(self.memory, batch_size)
       

        for state, action, reward, state_, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma* np.argmax(self.model.predict(state_)[0])
            # optimier
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self):
        self.model = load_model('my_model.h5')

    def save(self):
        self.model.save('my_model.h5')
