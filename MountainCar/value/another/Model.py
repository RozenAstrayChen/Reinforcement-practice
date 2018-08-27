import numpy as np
import tensorflow as tf
import tensorflow.logging as logging
import random


class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        # input layer
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        # output layer
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # hidden layer
        fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        # loss function
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        # back propagation
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()
    # simply returns the output of the networ
    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={
            self._states: state.reshape(1, self._num_states)
        })

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

    def print_num_of_total_parameters(self):
        pass

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self,sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)
    # sample returns a random selection of no_samples in length
    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)