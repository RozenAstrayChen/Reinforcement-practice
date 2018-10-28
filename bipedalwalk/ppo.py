import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 3001
#EP_LEN = 2048
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 64
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 24, 4
#S_DIM, A_DIM = 3, 1
epsilon = 0.2
'''
I think neural need using three full connect layer
first layer 24 -> 64
second layer 64 -> 128
third layer 64 -> action,1
'''


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrian_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sampe_op = tf.squeeze(
                pi.sample(1),
                axis=0  # choosing action
            )
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [
                oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)
            ]
        # loss
        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv

            self.aloss = -tf.reduce_mean(
                tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - epsilon, 1 + epsilon) *
                    self.tfadv))
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        #adv = (adv - adv.mean()) / adv.std()
        # update actor
        [
            self.sess.run(self.atrain_op, {
                self.tfs: s,
                self.tfa: a,
                self.tfadv: adv
            }) for _ in range(A_UPDATE_STEPS)
        ]
        # update critic
        [
            self.sess.run(self.ctrian_op, {
                self.tfs: s,
                self.tfdc_r: r
            }) for _ in range(C_UPDATE_STEPS)
        ]

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sampe_op, {self.tfs: s})[0]
        return a

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def save_model(self):
        save_path = self.saver.save(self.sess, "save/bike.ckpt")

    def load_model(self):
        self.saver.restore(self.sess, "save/bike.ckpt")

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(
                self.tfs, 100, tf.nn.relu, trainable=trainable)
            # output
            mu = 2 * tf.layers.dense(
                l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(
                l1, A_DIM, tf.nn.softplus, trainable=trainable)

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return norm_dist, params


watch_flag = True
env = gym.make('BipedalWalker-v2').unwrapped
#env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
all_ep_r = []
ppo.load_model()

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):  # in one episode
        if watch_flag:
            env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append(r)  # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(
                discounted_r)[:, np.newaxis]

            ppo.update(bs, ba, br)

    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(ep_r)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
    )
    if watch_flag == False:
        if ep % 500 == 0:
            print('touch save!')
            ppo.save_model()

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()
