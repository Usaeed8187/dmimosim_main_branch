
import numpy as np
import tensorflow as tf


# Deep Q Network off-policy
class basicDRQN:
    def __init__(
            self,
            su_index,
            n_actions,
            n_features,
            reward_decay=0.9,
            e_greedy=0.9,
            memory_size=300,
            lr=0.01,
            random_seed=1
    ):
        ###  
        raise RuntimeError('This  class is not implemented yet. It was supposed to be written using Keras but has not been done yet')
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = memory_size
        self.epsilon = e_greedy

        # initialize learning rate
        self.lr = lr

        self.output_type = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # build net
        self._build_net(su_index, random_seed)

        # replace the weights in target_net with the weights in eval_net
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net' + str(su_index))
        e_train_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_train' + str(su_index))
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net' + str(su_index))
        self.target_replace_op = [tf.assign(t, e_train) for t, e_train in zip(t_params, e_train_params)]
        self.eval_replace_op = [tf.assign(e, e_train) for e, e_train in zip(e_params, e_train_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



        self.cost_his = []

    def _build_net(self, su_index, random_seed=1):
        np.random.seed(random_seed)
        tf.random.set_random_seed(random_seed)

        self.nInternalUnits = 64  # 32
        self.time_step = 40   # 40, 50
        #self.nForgetPoints = 50  #

        self.s = tf.placeholder(tf.float32, [None, 1, self.n_features])  # input
        self.sReshape = tf.reshape(self.s, [-1, self.n_features])

        self.s_train = tf.placeholder(tf.float32, [None, self.time_step, self.n_features])  # input
        self.sReshape_train = tf.reshape(self.s_train, [-1, self.n_features])
        self.q_target_train = tf.placeholder(tf.float32, [None, self.time_step, self.n_actions])  # for calculating loss

        # ------------------ build evaluate_net ------------------
        self.final_s1_eval = []
        with tf.variable_scope('eval_net' + str(su_index)):
            self.rnnCell1_eval = tf.nn.rnn_cell.BasicRNNCell(num_units=self.nInternalUnits)
            self.init_s1_eval = self.rnnCell1_eval.zero_state(batch_size=1, dtype=tf.float32)
            self.outs3D1_eval, self.hidden_s1_eval = tf.nn.dynamic_rnn(self.rnnCell1_eval, self.s,
                                                                       initial_state=self.init_s1_eval,
                                                                       time_major=False)

            self.outs2D1_eval = tf.reshape(self.outs3D1_eval, [-1, self.nInternalUnits])
            if self.output_type == 1:
                self.concat_eval = tf.concat([self.outs2D1_eval, self.sReshape], axis=1)
                self.netOuts2D_eval = tf.layers.dense(self.concat_eval, self.n_actions)
            else:
                self.netOuts2D_eval = tf.layers.dense(self.outs2D1_eval, self.n_actions)

            self.q_eval = tf.reshape(self.netOuts2D_eval, [-1, 1, self.n_actions])

        with tf.variable_scope('eval_net_train' + str(su_index)):
            self.rnnCell1_eval_train = tf.nn.rnn_cell.BasicRNNCell(num_units=self.nInternalUnits)
            self.init_s1_eval_train = self.rnnCell1_eval_train.zero_state(batch_size=1, dtype=tf.float32)
            self.outs3D1_eval_train, self.hidden_s1_eval_train = tf.nn.dynamic_rnn(self.rnnCell1_eval_train, self.s_train,
                                                             initial_state=self.init_s1_eval_train,
                                                             time_major=False)

            self.outs2D1_eval_train = tf.reshape(self.outs3D1_eval_train, [-1, self.nInternalUnits])
            if self.output_type == 1:
                self.concat_eval_train = tf.concat([self.outs2D1_eval_train, self.sReshape_train], axis=1)
                self.netOuts2D_eval_train = tf.layers.dense(self.concat_eval_train, self.n_actions)
            else:
                self.netOuts2D_eval_train = tf.layers.dense(self.outs2D1_eval_train, self.n_actions)

            self.q_eval_train = tf.reshape(self.netOuts2D_eval_train, [-1, self.time_step, self.n_actions])

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target_train, self.q_eval_train))
        self.lr_eval_net = tf.placeholder(tf.float32, shape=[])
        self._train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr_eval_net).minimize(self.loss)

        self.s_ = tf.placeholder(tf.float32, [None, self.time_step, self.n_features])  # input
        self.sReshape_ = tf.reshape(self.s_, [-1, self.n_features])
        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net' + str(su_index)):
            self.rnnCell1_target = tf.nn.rnn_cell.BasicRNNCell(num_units=self.nInternalUnits)
            self.init_s1_target = self.rnnCell1_target.zero_state(batch_size=1, dtype=tf.float32)
            self.outs3D1_target, self.hidden_s1_target = tf.nn.dynamic_rnn(self.rnnCell1_target, self.s_,
                                                                   initial_state=self.init_s1_target,
                                                                   time_major=False)

            self.outs2D1_target = tf.reshape(self.outs3D1_target, [-1, self.nInternalUnits])
            if self.output_type == 1:
                self.concat_target = tf.concat([self.outs2D1_target, self.sReshape_], axis=1)
                self.netOuts2D_target = tf.layers.dense(self.concat_target, self.n_actions)
            else:
                self.netOuts2D_target = tf.layers.dense(self.outs2D1_target, self.n_actions)

            self.q_next = tf.reshape(self.netOuts2D_target, [-1, self.time_step, self.n_actions])

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        self.memory_counter = self.memory_counter % self.memory_size
        self.memory[self.memory_counter, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, np.newaxis, :]

        # forward feed the observation and get q value for every actions
        if (self.final_s1_eval == []):
            #zero_state_eval = np.zeros((1, self.nInternalUnits))
            actions_value, self.final_s1_eval = self.sess.run([self.q_eval, self.hidden_s1_eval],
                                                              feed_dict={self.s: observation})
        else:
            actions_value, self.final_s1_eval, concat_eval = self.sess.run([self.q_eval, self.hidden_s1_eval, self.hidden_s1_eval],
                                                              feed_dict={self.s: observation,
                                                              self.init_s1_eval: self.final_s1_eval})
            #actions_value, self.final_s1_eval, concat_eval = self.sess.run([self.q_eval, self.hidden_s1_eval, self.outs2D1_eval], feed_dict = {self.s: observation})
            #print(actions_value.reshape(-1))
            #print(concat_eval[0, -10:])

        actions_value = np.reshape(actions_value, -1)

        if np.random.uniform() < self.epsilon:
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action


    def learn_new(self, episode_size, step, method):

        batch = self.memory
        training_batch_size = self.time_step
        training_iteration = 100  # 200
        replace_target_iter = 25  # 50

        index_episode = np.random.randint(self.memory_counter / episode_size)


        for i in range(training_iteration):
            if (step == episode_size - 1):
                index_start = np.random.randint(episode_size * index_episode,
                                                episode_size * (index_episode + 1) - training_batch_size + 1)
            else:
                index_start = np.random.randint(episode_size * index_episode,
                                                episode_size * (index_episode + 1) - training_batch_size + 1)
            index_train = np.arange(index_start, index_start+training_batch_size)

            batch_memory = batch[index_train]
            s_train = batch_memory[:, :self.n_features]
            s_train = np.reshape(s_train, (1, training_batch_size, self.n_features))
            #zero_state_eval_train = np.zeros((1, self.nInternalUnits))
            q_eval = self.sess.run([self.q_eval_train], feed_dict={self.s_train: s_train})
            q_eval = np.reshape(q_eval, (training_batch_size, self.n_actions))

            s_ = batch_memory[:, -self.n_features:]
            s_ = np.reshape(s_, (1, training_batch_size, self.n_features))
            #zero_state_target = np.zeros((1, self.nInternalUnits))
            q_next = self.sess.run([self.q_next], feed_dict={self.s_: s_})
            q_next = np.reshape(q_next, (training_batch_size, self.n_actions))

            if (method == 'double'):
                next_action = np.argmax(q_next, axis=1)

            # change q_target w.r.t q_eval's action
            q_target_train = q_eval.copy()

            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:, self.n_features + 1]

            if (method == 'normal'):
                next_q_value = self.gamma * np.max(q_next, axis=1)
                for index in range(len(eval_act_index)):
                    q_target_train[index, eval_act_index[index]] = reward[index] + next_q_value[index]

            elif (method == 'double'):
                for index in range(len(eval_act_index)):
                    q_target_train[index, eval_act_index[index]] = reward[index] + \
                                                             self.gamma * q_next[index, next_action[index]]

            q_target_train = np.reshape(q_target_train, (1, training_batch_size, self.n_actions))
            # train eval network
            _ = self.sess.run([self._train_op], feed_dict={self.s_train: s_train,
                                                           self.q_target_train: q_target_train,
                                                           self.lr_eval_net: self.lr})

            if ((i+1) % replace_target_iter == 0):
                self.sess.run(self.target_replace_op)

        # Set the eval_net to eval_train after training
        self.sess.run(self.eval_replace_op)

        # Initialize the hidden state of eval_net
        self.final_s1_eval = []
        '''
        self.final_s1_eval = []
        for i in range(self.nForgetPoints):
            observation = batch[- self.nForgetPoints + i, :self.n_features]
            observation = observation[np.newaxis, np.newaxis, :]
            if (self.final_s1_eval == []):
                actions_value, self.final_s1_eval = self.sess.run([self.q_eval, self.hidden_s1_eval],
                                                                  feed_dict={self.s: observation})
            else:
                actions_value, self.final_s1_eval = self.sess.run([self.q_eval, self.hidden_s1_eval],
                                                                  feed_dict={self.s: observation,
                                                                             self.init_s1_eval: self.final_s1_eval})
        '''
    def update_lr(self, lr):
        self.lr = lr

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()