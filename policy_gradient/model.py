import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops


class PolicyGradient:
    def __init__(self, n_x, n_y, learning_rate=0.01, reward_decay=0.95):
        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        self.build_network()
        self.sess = tf.Session()

        tf.summary.FileWriter("logs/", self.sess.graph)

    def store_transition(self, s, a, r):
        """
            Store play memory training
        :param s: observation
        :param a: action
        :param r: reward
        :return:
        """
        # reshape observation to (1, num_features)

        self.episode_observations.append(s)
        self.episode_actions.append(a)
        self.episode_rewards.append(r)

    def choose_action(self, observation):
        """
            Choose best action
        :param observation:
        :return:
        """
        observation = observation[np.newaxis, :]

        # Run forward propagation

