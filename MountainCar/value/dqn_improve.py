import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

model_name = 'my_model.h5'

class DQNAgent:
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.batch_size = 32
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125


        # we need build two model
        self.model  =self.create_model()
        # 'hack' implemented by DeepMind to imporve convergence
        self.target_model = self.create_model()
    
    def create_model(self):
        model   = Sequential()
        
        model.add(Dense(24, input_dim=self.state_size, 
            activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    '''Rember function
    '''
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])



    ''' replay function
    '''
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        for state, action, reward, new_state, done in samples:
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_futrue = max(
                    self.target_model.predict(new_state)[0])
                target[0][action] = reward + self.gamma*(q_futrue)

            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()

        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    '''action function
    '''
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def save_model(self, fn):
        self.target_model.save(fn)
    def load(self, fn):
        self.model = load_model(model_name)
        self.target_model = load_model(model_name)


