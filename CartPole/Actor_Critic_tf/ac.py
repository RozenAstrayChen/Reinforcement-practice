import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 1000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n




class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None,
                                       "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
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

            self.acts_prob = tf.layers.dense(
                inputs=l2,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0.,
                                                                .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob')

        with tf.variable_scope('exp_v'):
            self.log_prob = tf.log(self.acts_prob[0, self.a])
            #self.log_prob = self.acts_prob
            self.exp_v = tf.reduce_mean(
                self.log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(
                -self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        #print('s = ', s.shape, '\na = ', a, '\ntd = ', td)
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v, log = self.sess.run([self.train_op, self.exp_v, self.log_prob], feed_dict)
        #print('exp_v = ', exp_v)

        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob,
                              {self.s: s})  # get probabilities for all actions
        return np.random.choice(
            np.arange(probs.shape[1]), p=probs.ravel())  # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=32,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0.,
                                                                .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1')

            l2 = tf.layers.dense(
                inputs=l1,
                units=32,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0.,
                                                                .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2')

            self.v = tf.layers.dense(
                inputs=l2,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0.,
                                                                .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V')

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(
                self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _, loss = self.sess.run([self.td_error, self.train_op, self.loss], {
            self.s: s,
            self.v_: v_,
            self.r: r
        })
        return td_error, loss[0,0]


sess = tf.Session()
collect_rewards = []
actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(
    sess, n_features=N_F, lr=LR_C
)  # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("origin_logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error, c_loss = critic.learn(
            s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        exp = actor.learn(s, a,
                    td_error)  # true_gradient = grad[logPi(s,a) * td_error]
        

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)
            collect_rewards.append(ep_rs_sum)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

            if i_episode > 400:
                RENDER = True  # rendering
            print(-exp + c_loss)
            print("episode:", i_episode, "  reward:", int(ep_rs_sum))
            break

plt.figure(2)
plt.clf()
plt.title('Training...')
plt.xlabel('Episode')
plt.ylabel('Origin')
plt.plot(collect_rewards)
#plt.show()
plt.savefig('./improve.jpg')

