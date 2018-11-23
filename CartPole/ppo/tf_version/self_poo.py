import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from rollout import Rollout


class Agent():
    def __init__(self, sess, env, a_space, s_space):
        self.env = env
        self.sess = sess
        self.a_space = a_space
        self.s_space = s_space
        self.lr = 2e-4
        self.train_episodes = 5000
        self.gamma = 0.99
        self.epsilon = 0.2
        self.horizen = 128
        self.batch = 32
        self.rollout = Rollout(self.batch)
        self.entropy_coff = 0.02
        self.value_coff = 1

        # input tf variable
        self._init_input()
        # Actor-Critic output
        self._init_net_out()
        # operation
        self._init_op()

        self.sess.run(tf.global_variables_initializer())

    def _init_input(self, ):
        with tf.variable_scope('input'):
            self.s_in = tf.placeholder(
                tf.float32, [None, self.s_space], name='s_in')
            self.a_in = tf.placeholder(tf.int32, [None], name='a_in')
            self.r_in = tf.placeholder(tf.float32, [None, 1], name='r_in')
            self.adv_in = tf.placeholder(tf.float32, [None], name='adv_in')

    def _init_net_out(self, ):
        self.v, self.a_prob = self._init_ac_net('theta')
        self.v_old, self.a_prob_old = self._init_ac_net(
            'theta_old', trainable=False)

    def _init_ac_net(self, scope, trainable=True):
        with tf.variable_scope(scope):
            # Kernel initializer.
            w_initializer = tf.random_normal_initializer(0.0, 0.01)
            # first dense
            f_dense = tf.layers.dense(
                self.s_in,
                32,
                tf.nn.relu,
                trainable=trainable,
                kernel_initializer=w_initializer)
            # second dense
            s_dense = tf.layers.dense(
                f_dense,
                32,
                tf.nn.relu,
                trainable=trainable,
                kernel_initializer=w_initializer)
            # critic output
            v = tf.layers.dense(
                s_dense,
                1,
                trainable=trainable,
                kernel_initializer=w_initializer)
            # actor output
            a_prob = tf.layers.dense(
                s_dense,
                self.a_space,
                tf.nn.softmax,
                trainable=trainable,
                kernel_initializer=w_initializer)

            return v, a_prob

    def _init_op(self, ):
        with tf.variable_scope('update_params'):
            params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='theta')
            params_old = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='theta_old')
            self.update_params = [
                tf.assign(old, now) for old, now in zip(params_old, params)
            ]
        '''
        with tf.variable_scope('advantage'):
            self.adv = self.r_in - self.v
        '''

        with tf.variable_scope('loss'):

            #self.c_loss = tf.losses.mean_squared_error(self.r_in, self.v)

            # Actor Loss
            a_indices = tf.stack(
                [tf.range(tf.shape(self.a_in)[0], dtype=tf.int32), self.a_in],
                axis=1)
            # theta
            pi_prob = tf.gather_nd(params=self.a_prob, indices=a_indices)

            # old_theta
            pi_old_prob = tf.gather_nd(
                params=self.a_prob_old, indices=a_indices)

            # surrogate 1
            ratio = pi_prob / pi_old_prob

            surr1 = ratio * self.adv_in

            # surrogate 2
            surr2 = tf.clip_by_value(ratio, 1. - self.epsilon,
                                     1. + self.epsilon) * self.adv_in

            # estimate
            self.a_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            # Critic Loss
            self.c_loss = tf.reduce_mean(tf.square(self.adv_in))
            # dist entropy
            self.entropy = tf.reduce_mean(
                self.a_prob * tf.log(self.a_prob))  # encourage exploration

            self.loss = self.a_loss + self.c_loss - self.entropy * self.entropy_coff

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(
                self.loss)

    '''[summary]
    predict action with state
    Returns:
        [type] -- [description]
        v : critic output
        a : actor output
    '''

    def get_v(self, s):
        v = self.sess.run([self.v], feed_dict={self.s_in: s})
        return v[0][0]

    def choose_action(self, s):
        a_prob = self.sess.run(self.a_prob, feed_dict={self.s_in: [s]})
        a = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())

        return a

    def calculate_returns(self, rewards, dones, last_value, gamma=0.99):
        rewards = np.array(rewards)
        dones = np.array(dones)
        # create nparray
        returns = np.zeros(rewards.shape[0] + 1)
        returns[-1] = last_value
        dones = 1 - dones
        for i in reversed(range(rewards.shape[0])):
            returns[i] = gamma * returns[i + 1] * dones[i] + rewards[i]

        return returns

    '''
    plt loss
    '''

    def plot_loss(self, reward, a_loss, c_loss, entropy):

        plt.figure(1)
        plt.clf()
        plt.title('Loss')
        plt.xlabel('Episode*10')
        plt.ylabel('Duration')
        plt.plot()
        plt.plot(reward, color='red', label='reward')
        plt.plot(a_loss, color='green', label='a_loss')
        plt.plot(c_loss, color='blue', label='c_loss')
        plt.plot(entropy, color='black', label='entropy')
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated

    def plot_saved(self, reward, a_loss, c_loss, entropy):
        plt.figure(1)
        plt.clf()
        plt.title('Loss')
        plt.xlabel('Episode*10')
        plt.ylabel('Duration')
        plt.plot()
        plt.plot(reward, color='red', label='reward')
        plt.plot(a_loss, color='green', label='a_loss')
        plt.plot(c_loss, color='blue', label='c_loss')
        plt.plot(entropy, color='black', label='entropy')
        name = './ppo' + '_loss' + '.jpg'
        plt.savefig(name)

    def train(self, memory):
        # First, update old theta
        self.sess.run(self.update_params)
        # Second calculate advantage
        s = np.array(memory.s)
        returns = np.array(memory.r)
        #newaxis
        returns = returns[:, np.newaxis]
        predicts = self.get_v(s)
        #print('ret', returns.shape, '\tand one is', returns[0])
        #print('predicts', predicts.shape, '\tand one is', predicts[0])
        adv = returns - predicts
        adv = adv.ravel()
        adv = (adv - adv.mean()) / adv.std()

        memory.adv_replace(adv)
        # update N times
        for _ in range(5):
            # sample data
            s, a, adv = memory.sample()
            adv = adv.ravel()

            dict = {self.s_in: s, self.a_in: a, self.adv_in: adv}

            _, loss, a_loss, c_loss, entropy = self.sess.run(
                [
                    self.optimizer, self.loss, self.a_loss, self.c_loss,
                    self.entropy
                ],
                feed_dict=dict)

            #self.sess.run([self.optimizer], feed_dict=dict)
    def run(self):
        for episode in range(self.train_episodes):
            s = self.env.reset()
            done = False
            for t in range(self.horizen):
                if done:
                    s = self.env.reset()
                
                a = self.choose_action(s)
                n_s, r, done, _ = self.env.step(a)
                if done:
                    r = -5
                self.rollout.append(s, a, r, n_s, done)
                s = n_s

            if done:
                
                last_value = 0
            else:
                s = s[np.newaxis, :]
                last_value = self.get_v(s)

            returns = self.calculate_returns(self.rollout.r, self.rollout.done,
                                             last_value)
            self.rollout.r = returns[:-1]
            self.train(self.rollout)
            self.rollout.flush()
            '''
            learning 
            '''

            if episode % 50 == 0:
                
                r_steps = 0
                s = self.env.reset()
                while True:
                    self.env.render()
                    a = self.choose_action(s)
                    n_s, r, done, _ = self.env.step(a)
                    r_steps += r
                    s = n_s
                    if done:
                        print('episode = {} ; get_reward = {}'.format(
                            episode, r_steps))
                        break


# Make env.
env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')
env.seed(1)
env = env.unwrapped
# Init session.
sess = tf.Session()
agent = Agent(sess, env, env.action_space.n, env.observation_space.shape[0])
agent.run()
