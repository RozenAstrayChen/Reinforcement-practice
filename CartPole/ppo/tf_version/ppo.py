# coding=utf-8

import tensorflow as tf
import numpy as np
import gym
import sys

sys.path.append('..')


class Agent():

    def __init__(self, session, env, a_space, s_space):
        self.env = env
        self.session = session
        self.a_space = a_space
        self.s_space = s_space
        self.learning_rate = 2e-4
        self.mode = 'train'
        self.train_episodes = 40000
        self.gamma = 0.9

        self._init_input()
        self._init_nn()
        self._init_op()
        #self._init_saver()

        self.a_buffer = []
        self.s_buffer = []
        self.r_buffer = []
        self.a_p_r_buffer = []

        self.session.run(tf.global_variables_initializer())

    def _init_input(self, *args):
        with tf.variable_scope('input'):
            self.s = tf.placeholder(tf.float32, [None, self.s_space], name='s')
            self.a = tf.placeholder(tf.int32, [None, ], name='a')
            self.r = tf.placeholder(tf.float32, [None, ], name='r')
            self.adv = tf.placeholder(tf.float32, [None, ], name='adv')
            self.a_p_r = tf.placeholder(tf.float32, [None, ], name='a_p_r')

    def _init_nn(self, *args):
        self.advantage, self.value = self._init_critic_net('critic_net')
        self.a_prob_eval, self.a_logits_eval = self._init_actor_net('eval_actor_net')
        self.a_prob_target, self.a_logits_target = self._init_actor_net('target_actor_net', trainable=False)

    def _init_op(self):
        with tf.variable_scope('critic_loss_func'):
            # loss func.
            self.c_loss_func = tf.losses.mean_squared_error(labels=self.r, predictions=self.value)
        with tf.variable_scope('critic_optimizer'):
            # critic optimizer.
            self.c_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.c_loss_func)
        with tf.variable_scope('update_target_actor_net'):
            # Get eval w, b.
            params_e = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_actor_net')
            params_t = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor_net')
            self.update_target_a_op = [tf.assign(t, e) for t, e in zip(params_t, params_e)]
        with tf.variable_scope('actor_loss_func'):
            '''
            # one hot a.
            a_one_hot = tf.one_hot(self.a, self.a_space)
            # cross entropy.
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=a_one_hot, logits=self.a_logits_eval)
            # loss func.
            self.a_loss_func = tf.reduce_mean(cross_entropy * self.adv * self.a_p_r)
            '''
            a_indices = tf.stack(
                [tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a],
                axis=1)

            new_policy_prob = tf.gather_nd(
                params=self.a_prob_eval, indices=a_indices)  # shape=(None, )

            old_policy_prob = tf.gather_nd(
                params=self.a_prob_target, indices=a_indices)  # shape=(None, )

            ratio = new_policy_prob / old_policy_prob

            surr = ratio * self.adv  # surrogate loss
            EPSILON = 0.2
            # clipped surrogate objective
            self.a_loss_func = -tf.reduce_mean(
                tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) *
                    self.adv))
                    
        with tf.variable_scope('actor_optimizer'):
            self.a_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.a_loss_func)

    def _init_actor_net(self, scope, trainable=True):
        with tf.variable_scope(scope):
            # Kernel initializer.
            w_initializer = tf.random_normal_initializer(0.0, 0.01)
            # First dense.
            f_dense = tf.layers.dense(self.s, 32, tf.nn.relu, trainable=trainable, kernel_initializer=w_initializer)
            # Second dense.
            s_dense = tf.layers.dense(f_dense, 32, tf.nn.relu, trainable=trainable, kernel_initializer=w_initializer)
            # Action logits.
            a_logits = tf.layers.dense(s_dense, self.a_space, trainable=trainable, kernel_initializer=w_initializer)
            # Action prob.
            a_prob = tf.nn.softmax(a_logits)
            return a_prob, a_logits

    def _init_critic_net(self, scope):
        with tf.variable_scope(scope):
            # Kernel initializer.
            w_initializer = tf.random_normal_initializer(0.0, 0.01)
            # First dense.
            f_dense = tf.layers.dense(self.s, 64, tf.nn.relu, kernel_initializer=w_initializer)
            # Value.
            value = tf.layers.dense(f_dense, 1)
            value = tf.reshape(value, [-1, ])
            # Advantage.
            advantage = self.r - value
            return advantage, value

    def predict(self, s):
        # Calculate a eval prob.
        a_prob_eval, a_prob_target = self.session.run([self.a_prob_eval, self.a_prob_target], {self.s: [s]})
        # Calculate action prob ratio between eval and target.
        a_p_r = np.max(a_prob_eval) / np.max(a_prob_target)
        self.a_p_r_buffer.append(a_p_r)
        return np.random.choice(range(a_prob_eval.shape[1]), p=a_prob_eval.ravel())

    def snapshot(self, s, a, r, _):
        self.a_buffer.append(a)
        self.s_buffer.append(s)
        self.r_buffer.append(r)

    def train(self):
        self.session.run(self.update_target_a_op)
        # Copy r_buffer
        r_buffer = self.r_buffer
        # Init r_tau
        r_tau = 0
        # Calculate r_tau
        for index in reversed(range(0, len(r_buffer))):
            r_tau = r_tau * self.gamma + r_buffer[index]
            self.r_buffer[index] = r_tau
        # Calculate adv.
        print('r_buffer : ', self.r_buffer)
        adv_buffer = self.session.run(self.advantage, {self.s: self.s_buffer, self.r: self.r_buffer})
        print('adv : ', adv_buffer)
        # Minimize loss.
        [
            self.session.run([self.a_optimizer, self.c_optimizer], {
                self.adv: adv_buffer,
                self.s: self.s_buffer,
                self.a: self.a_buffer,
                self.r: self.r_buffer,
                self.a_p_r: self.a_p_r_buffer,
            }) for _ in range(5)
        ]
        self.s_buffer = []
        self.a_buffer = []
        self.r_buffer = []
        self.a_p_r_buffer = []

    def run(self):
        if self.mode == 'train':
            for episode in range(self.train_episodes):
                s, r_episode = self.env.reset(), 0
                while True:
                    if episode > 200:
                        self.env.render()
                    a = self.predict(s)
                    s_n, r, done, _ = self.env.step(a)
                    if done:
                        r = -5
                    r_episode += r
                    self.snapshot(s, a, r, s_n)
                    s = s_n
                    if done:
                        break
                self.train()
                if episode % 25 == 0:
                    print('Episode: {} | Rewards: {}'.format(episode, r_episode))
                    #self.save()
        else:
            for episode in range(self.eval_episodes):
                s, r_episode = self.env.reset()
                while True:
                    a = self.predict(s)
                    s_n, r, done, _ = self.env.step(a)
                    r_episode += r
                    s = s_n
                    if done:
                        break



# Make env.
env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')
env.seed(1)
env = env.unwrapped
# Init session.
session = tf.Session()
# Init agent.
agent = Agent(session, env, env.action_space.n, env.observation_space.shape[0])
agent.run()