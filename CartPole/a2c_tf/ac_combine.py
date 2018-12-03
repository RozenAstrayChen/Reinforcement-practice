import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 1000
MAX_EP_STEPS = 1000  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR = 0.001  # learning rate for actor critic

env = gym.make('CartPole-v0')
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class AC_net(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s_in = tf.placeholder(tf.float32, [1, n_features], 'state')
        self.a_in = tf.placeholder(tf.int32, None, "act")
        self.r_in = tf.placeholder(tf.float32, None, 'r')
        self.v_in_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.td_error_in = tf.placeholder(tf.float32, None, 'td_error')

        with tf.variable_scope('AC'):
            w_initializer = tf.random_normal_initializer(0.0, 0.01)
            l1 = tf.layers.dense(
                inputs=self.s_in,
                units=32,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.,
                                                                .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1')

            l2 = tf.layers.dense(
                inputs=l1,
                units=32,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.,
                                                                .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2')
            # actor
            self.acts_prob = tf.layers.dense(
                inputs=l2,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

            # critic
            self.v = tf.layers.dense(
                inputs=l2,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )
        '''
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r_in + GAMMA * self.v_in_ - self.v
            
        with tf.variable_scope('c_loss'):
            #c_loss = tf.square(self.td_error)
            self.c_loss = tf.reduce_mean(tf.square(self.td_error))
        
        with tf.variable_scope('a_loss'):
            log_prob = tf.log(self.acts_prob[0, self.a_in])
            self.a_loss = -tf.reduce_mean(log_prob * self.td_error)

            entropy = tf.reduce_mean(
                self.acts_prob * tf.log(self.acts_prob))  # encourage exploration
                  
        with tf.variable_scope('loss'):
            #self.loss = a_loss + c_loss - 0.02 * entropy
            self.loss = self.a_loss + self.c_loss
        '''
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r_in + GAMMA * self.v_in_ - self.v

        with tf.variable_scope('loss'):   
            log_prob = tf.log(self.acts_prob[0, self.a_in])
            a_loss = -tf.reduce_mean(log_prob * self.td_error)

            entropy = tf.reduce_mean(
                self.acts_prob * tf.log(self.acts_prob))  # encourage exploration
                  
            c_loss = tf.reduce_mean(tf.square(self.td_error))
            #self.loss = a_loss + c_loss - 0.02 * entropy
            self.loss = a_loss + c_loss - 0.02 * entropy

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(
            self.acts_prob, {self.s_in: s})  # get probabilities for all actions

        return np.random.choice(
            np.arange(probs.shape[1]), p=probs.ravel())  # return a int


    def learn(self, s, a, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        # evaluate next v
        v_ = self.sess.run(self.v, {self.s_in: s_})
        # caculate actor & critic
        feed = {    
            self.s_in: s,
            self.a_in: a,
            self.v_in_: v_,
            self.r_in: r
        }
        _, loss = self.sess.run([self.train_op, self.loss], feed)
        #print('a', aloss)
        #print('c', closs)
        #print('l', loss)
        
        return loss
    
        

sess = tf.Session()
collect_rewards = []
collect_losses = []
ac = AC_net(sess, n_features=N_F, n_actions=N_A, lr=LR)
sess.run(tf.global_variables_initializer())


if OUTPUT_GRAPH:
    tf.summary.FileWriter("combine_logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = 0
    
    while True:
        if RENDER: env.render()
        a = ac.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r += r
        loss = ac.learn(s, a, r, s_)  # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            collect_rewards.append(track_r)
            collect_losses.append(loss)
            '''
            if i_episode > 400:
                RENDER = True  # rendering
            '''

            print("episode:", i_episode, "  reward:", track_r)
            print('loss:', loss)
            break

plt.figure(2)
plt.clf()
plt.title('Training...')
plt.xlabel('Episode')
plt.ylabel('Origin')
plt.plot(collect_rewards, color='blue', label='reward')
plt.plot(collect_losses, color='red', label='loss')
plt.show()
plt.savefig('./ac_combine.jpg')
