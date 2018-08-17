import tensorflow as tf
import numpy as np


class DeepQNetwork:
    # build neural network
    def _build_net(self):
        '''
        -----create eval neural network, and promotion paramters-----
        '''
        # recive observation
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        # recive q_target which will get after calcuate
        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name='Q_target')
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) is paramters when update target_net
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1) # config of layers

            # First layers of eval_net. collections which paramters used when update
            with tf.variable_scope('l1'):
                # weight
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1],
                    initializer=w_initializer,
                    collections=c_names)
                # bias
                b1 = tf.get_variable(
                    'b1', [1, n_l1],
                    initializer=b_initializer,
                    collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # Second layers of eval_net. collections which paramters used when update
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w1', [n_l1, self.n_actions],
                    initializer=w_initializer,
                    collections=c_names)
                b2 = tf.get_variable(
                    'b1', [1, self.n_actions],
                    initializer=b_initializer,
                    collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
        '''
        loss function
        '''
        with tf.variable_scope('loss'):  # error  from q_target and q_eval
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval))
        '''
        back propagation
        '''
        with tf.variable_scope('train'):  # gradinet descent
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(
                self.loss)
        '''
        ------ create target neural network, provide target Q----
        '''
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='s_')  # 接收下个 observation
        with tf.variable_scope('target_net'):
            # c_names(collections_names) is paramters when update target_net
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # First layers of target_net. collections which paramters used when update
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1],
                    initializer=w_initializer,
                    collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1],
                    initializer=b_initializer,
                    collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # Second layers of target_net. collections which paramters used when update
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions],
                    initializer=w_initializer,
                    collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions],
                    initializer=b_initializer,
                    collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter  # change target_net step
        self.memory_size = memory_size  # memory size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment  #epsilon num
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # Enable explore mode, and decrease expolore step

        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

        t_params = tf.get_collection(
            'target_net_parans')  # extract target_net params
        e_params = tf.get_collection(
            'eval_net_params')  # extract eval_net params
        # update target_net params
        self.replace_target_op = [
            tf.assign(t, e) for t, e in zip(t_params, e_params)
        ]

        self.sess = tf.Session()
        #oupt tensorboard fil
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        #record cost change after showing plot
        self.cost_hist = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # record vector of [s, a, r, s_]
        transition = np.hstack((s, [a, r], s_))
        # if over size , old memory will be replace from new one
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # replace
        self.memory_counter += 1

    def choose_action(self, observation):
        # collect all shape of observation (1, size_of_obs)

        if np.random.uniform() < self.epsilon:
            observation = observation.reshape(1, 2)
            # let eval_net neural network generate all action value, and select max one
            actions_value = self.sess.run(
                self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check target_net has been changed or not
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
        #select batch in population
        if self.memory_counter > self.memory_size:
            sampe_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sampe_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sampe_index, :]

        # get q_next (target_net genetate q ) and q_eval(eval_net g q)
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                self.s: batch_memory[:, :self.n_features]
            })
        # important
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        # Q learning
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(
            q_next, axis=1)
        # back propragation
        _, self.cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.q_target: q_target
            })
        # record cost error
        self.cost_hist.append(self.cost)
        #decrease epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_hist)), self.cost_hist)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def save(self):
        # 建立 saver 物件
        saver = tf.train.Saver()
        saver.save(self.sess, "model.ckpt")

    def load(self):
        saver = tf.train.Saver()
        #with tf.Session() as sess:
        saver.restore(self.sess, "model.ckpt")
