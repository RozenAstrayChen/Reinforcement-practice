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
        self.train_episodes = 10000
        self.gamma = 0.99
        self.epsilon = 0.2
        self.batch = 64
        self.rollout = Rollout()

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

        with tf.variable_scope('advatage'):
            self.adv = self.r_in - self.v

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
            self.entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5))  # encourage exploration

            self.loss = self.a_loss + 0.5 * (self.c_loss) + self.entropy * 0.1

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
        v = self.sess.run([self.v_old], feed_dict={self.s_in: [s]})
        return v[0][0]

    def choose_action(self, s):
        _, a_prob = self.sess.run(
            [self.v, self.a_prob], feed_dict={self.s_in: [s]})
        a = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())

        return a
    
    '''
    plt loss
    '''
    def plot_loss(self, reward, loss, a_loss, c_loss, entropy):


        plt.figure(1)
        plt.clf()
        plt.title('Loss')
        plt.xlabel('Episode*10')
        plt.ylabel('Duration')
        plt.plot()
        plt.plot(reward, color='yellow', label='reward')
        plt.plot(loss, color='red', label='loss')
        plt.plot(a_loss, color='green', label='a_loss')
        plt.plot(c_loss, color='blue', label='c_loss')
        plt.plot(entropy, color='black', label='entropy')
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated


    def train(self, ):
        #update old theta
        self.sess.run(self.update_params)

        s, a, r = self.rollout.sample()

        adv = self.sess.run(self.adv, {self.s_in: s, self.r_in: r})
        #adv = np.array(adv)
        adv = adv.ravel()
        #print('adv shape ', adv.shape)

        dict = {self.s_in: s, self.r_in: r, self.a_in: a, self.adv_in: adv}

        for _ in range(5):
            _ ,loss, a_loss, c_loss, entropy = self.sess.run(
                [
                    self.optimizer, self.loss, self.a_loss, self.c_loss,
                    self.entropy
                ],
                feed_dict=dict)
        
        self.rollout.clean()

        return loss, a_loss, c_loss, entropy

    def run(self):
        step = 0
        reward_collect = []
        loss_collect = []
        a_loss_collect = []
        c_loss_collect = []
        entropy_collect = []

        for episode in range(self.train_episodes):
            r_collect = []
            s = self.env.reset()
            while True:
                step += 1
                a = self.choose_action(s)
                n_s, r, done, _ = self.env.step(a)
                r_collect.append(r)

                self.rollout.append(s, a, r)
                s = n_s
                # Batch update
                if (step + 1) % self.batch == 0 or done:
                    v_s_ = self.get_v(n_s)
                    discount_r = []
                    for r in self.rollout.r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discount_r.append(v_s_)
                    discount_r.reverse()
                    # update
                    self.rollout.r = discount_r
                    loss, a_loss, c_loss, entropy = self.train()

                if done:
                    break
            '''
            learning 
            '''

            if episode % 50 == 0:
                print('episode : ', episode, 'reward :', sum(r_collect))
                reward_collect.append(sum(r_collect))
                loss_collect.append(loss)
                a_loss_collect.append(a_loss)
                c_loss_collect.append(c_loss)
                entropy_collect.append(entropy)
                self.plot_loss(reward_collect, loss_collect, a_loss_collect, c_loss_collect, entropy_collect)


                
            #print('loss : ', loss)
            if episode >= 5000:
                self.env.render()


# Make env.
env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')
env.seed(1)
env = env.unwrapped
# Init session.
sess = tf.Session()
agent = Agent(sess, env, env.action_space.n, env.observation_space.shape[0])
agent.run()
