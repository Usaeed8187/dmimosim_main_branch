import numpy as np
#from pyESN_online_new import ESN
#from pyDeepESN_online import ESN
#from pyDeepESN2_online import ESN
#from pyDeepESN3_online import ESN
from pyDeepWESN4_online import WESN #best
#from pyDeepESN5_online import ESN
#from pyDeepESN6_online import ESN
#from pyESN_online_decorr import ESN
import copy
import pickle
from pathlib import Path


# Deep Q Network off-policy
class DeepWESNQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            input_window_size,
            output_window_size,
            memory_size,
            n_layers,
            nInternalUnits = 64,
            reward_decay=0.9,
            e_greedy=0.9,
            min_epsilon=0.2,
            lr=0.01,
            random_seed=1,
            spectral_radius = 0.30
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_layers = n_layers
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.batch_size = memory_size
        self.max_epsilon = e_greedy
        self.min_epsilon = min_epsilon
        self.epsilon = self.min_epsilon
        self.spectral_radius = spectral_radius
        self.nInternalUnits = nInternalUnits

        # total learning step
        self.learn_step_counter = 0

        # initialize learning rate
        self.lr = lr

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # build net
        self._build_net(random_seed)

        self.cost_his = []

        self.training_batch_size = 40
        self.training_iteration = 100
        self.replace_target_iter = 25

        self.rng = np.random.RandomState(random_seed) # Line added by Ramin. Random number generator.

        

    def _build_net(self, random_seed):
        # ------------------ ESN parameters ------------------
        nInternalUnits = self.nInternalUnits #64 # 32, 128
        spectralRadius = self.spectral_radius
        inputScaling = 2 * np.ones(self.n_features)
        inputScaling = 1 * np.ones(self.n_features)
        #inputScaling = 1 * np.ones(self.n_features)
        inputShift = -1 * np.ones(self.n_features)
        inputShift = 0 * np.ones(self.n_features)
        #inputShift = 0 * np.ones(self.n_features)
        teacherScaling = 1 * np.ones(self.n_actions)
        teacherShift = 0 * np.ones(self.n_actions)
        self.nForgetPoints = 1  # 50

        # ------------------ build evaluate_net ------------------
        self.eval_net = WESN(n_inputs=self.n_features, n_outputs=self.n_actions, n_reservoir=nInternalUnits,
                            n_layers = self.n_layers, memory_size = self.memory_size,
                            input_window_length = self.input_window_size,
                            output_window_length = self.output_window_size,
                            spectral_radius=spectralRadius, sparsity=0, noise=0,
                            lr=self.lr,
                            input_shift=inputShift, input_scaling=inputScaling,
                            teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                            random_seed=random_seed)

        # ------------------ build target_net ------------------
        self.target_net = WESN(n_inputs=self.n_features, n_outputs=self.n_actions, n_reservoir=nInternalUnits,
                            n_layers = self.n_layers, memory_size = self.memory_size,
                            input_window_length = self.input_window_size,
                            output_window_length = self.output_window_size,
                            spectral_radius=spectralRadius, sparsity=0, noise=0,
                            lr=self.lr,
                            input_shift=inputShift, input_scaling=inputScaling,
                            teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                            random_seed=random_seed)

        self.target_net = copy.deepcopy(self.eval_net)
        #self.target_net.copy_net(self.eval_net)

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
        observation = observation[np.newaxis, :]

        # forward feed the observation and get q value for every actions
        actions_value = self.eval_net.predict(observation, 0, continuation=True)
        #print(actions_value)
        # if np.random.uniform() < self.epsilon:
        if self.rng.uniform() < self.epsilon: # Line added by Ramin
            action = np.argmax(actions_value)
        else:
            # action = np.random.randint(0, self.n_actions)
            action = self.rng.randint(0, self.n_actions)
            #action = 2 * np.random.randint(0, int(self.n_actions/2))
        return action

    def update_epsilon(self, current_step: int, total_steps: int) -> float:
        """Linearly anneal epsilon from ``min_epsilon`` to ``max_epsilon``.

        Args:
            current_step: The current episode or iteration index (0-based).
            total_steps: Total number of steps to complete the annealing schedule.

        Returns:
            The updated epsilon value.
        """

        if total_steps <= 0:
            self.epsilon = self.max_epsilon
            return self.epsilon

        clamped_step = max(0, min(current_step, total_steps))
        epsilon_span = self.max_epsilon - self.min_epsilon
        if epsilon_span <= 0:
            self.epsilon = self.max_epsilon
            return self.epsilon

        progress = clamped_step / float(total_steps)
        self.epsilon = min(self.max_epsilon, self.min_epsilon + epsilon_span * progress)
        return self.epsilon

    def activate_target_net(self, observation_):
        # to have batch dimension when feed into tf placeholder
        observation_ = observation_[np.newaxis, :]

        # forward feed the observation and get q value for every actions
        actions_value = self.target_net.predict(observation_, 0, continuation=True)


    def learn_new(self, episode_size, step, method):

        sequential = False
        batch = self.memory
        training_batch_size = self.training_batch_size  # 40, 50
        training_iteration = self.training_iteration  # 100, 200
        replace_target_iter = self.replace_target_iter  # 25, 50, 50
        

        if (step == episode_size - 1):
            # index_start = np.random.randint(episode_size * index_episode + self.nForgetPoints,
            #                                episode_size * (index_episode + 1) - training_batch_size + 1)
            index_train = np.arange(self.nForgetPoints, episode_size)
        else:
            # index_start = np.random.randint(episode_size * index_episode,
            #                                episode_size * (index_episode+1) - training_batch_size + 1)
            index_train = np.arange(0, episode_size)

        for i in range(training_iteration):
            if (sequential==True):
                if (step == episode_size - 1):
                    # index_start = np.random.randint(self.nForgetPoints,
                    #                                 episode_size - training_batch_size + 1)
                    index_start = self.rng.randint(self.nForgetPoints, # Line added by Ramin
                                                    episode_size - training_batch_size + 1)
                else:
                    # index_start = np.random.randint(0,
                    #                                 episode_size - training_batch_size + 1)
                    index_start = self.rng.randint(0, # Line added by Ramin
                                                    episode_size - training_batch_size + 1)
                index_train = np.arange(index_start, index_start+training_batch_size)
            else:
                # np.random.shuffle(index_train)
                self.rng.shuffle(index_train) # Line added by Ramin 
            batch_memory = batch[index_train[:training_batch_size]]
            #
            q_eval = self.eval_net.predict_training(index_train[:training_batch_size])
            q_next = self.target_net.predict_training(index_train[:training_batch_size])
            if (method == 'double'):
                q_next_action = self.eval_net.predict_training(index_train[:training_batch_size])
                next_action = np.argmax(q_next_action, axis = 1)


            # change q_target w.r.t q_eval's action
            q_target = q_eval.copy()

            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:, self.n_features + 1]

            if (method == 'normal'):
                next_q_value = self.gamma * np.max(q_next, axis=1)
                for index in range(len(eval_act_index)):
                    q_target[index, eval_act_index[index]] = reward[index] + next_q_value[index]

            elif (method == 'double'):
                for index in range(len(eval_act_index)):
                    q_target[index, eval_act_index[index]] = reward[index] + \
                                                             self.gamma * q_next[index, next_action[index]]

            # train eval network
            self.eval_net.fit(q_target, index_train[:training_batch_size])

            if ((i+1) % replace_target_iter == 0):
                self.target_net.W_out = copy.deepcopy(self.eval_net.W_out)
                #self.target_net.copy_net(self.eval_net)

        # Refresh the laststate
        self.eval_net.refresh_state()
        self.target_net.refresh_state()


    def update_lr(self, lr):
        self.eval_net.lr = lr
        self.target_net.lr = lr

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def save(self, path):
        """Persist network state to disk using pickle."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        init_kwargs = {
            "n_actions": self.n_actions,
            "n_features": self.n_features,
            "input_window_size": self.input_window_size,
            "output_window_size": self.output_window_size,
            "memory_size": self.memory_size,
            "n_layers": self.n_layers,
            "nInternalUnits": self.nInternalUnits,
            "reward_decay": self.gamma,
            "e_greedy": self.max_epsilon,
            "min_epsilon": self.min_epsilon,
            "lr": self.lr,
            "random_seed": None,
            "spectral_radius": self.spectral_radius,
        }

        data = {
            "init_kwargs": init_kwargs,
            "eval_net": self.eval_net,
            "target_net": self.target_net,
            "epsilon": self.epsilon,
            "max_epsilon": self.max_epsilon,
            "min_epsilon": self.min_epsilon,
            "memory": self.memory,
            "memory_counter": getattr(self, "memory_counter", 0),
            "learn_step_counter": self.learn_step_counter,
            "rng_state": self.rng.get_state(),
            "cost_his": self.cost_his,
            "training_batch_size": self.training_batch_size,
            "training_iteration": self.training_iteration,
            "replace_target_iter": self.replace_target_iter,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load a saved network from disk."""

        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        init_kwargs = data.get("init_kwargs", {})
        obj = cls(**init_kwargs)

        obj.eval_net = data.get("eval_net")
        obj.target_net = data.get("target_net")
        obj.epsilon = data.get("epsilon", obj.epsilon)
        obj.max_epsilon = data.get("max_epsilon", getattr(obj, "max_epsilon", obj.epsilon))
        obj.min_epsilon = data.get("min_epsilon", getattr(obj, "min_epsilon", 0.2))
        obj.memory = data.get("memory", obj.memory)
        obj.memory_counter = data.get("memory_counter", getattr(obj, "memory_counter", 0))
        obj.learn_step_counter = data.get("learn_step_counter", obj.learn_step_counter)
        obj.cost_his = data.get("cost_his", [])
        obj.training_batch_size = data.get("training_batch_size", obj.training_batch_size)
        obj.training_iteration = data.get("training_iteration", obj.training_iteration)
        obj.replace_target_iter = data.get("replace_target_iter", obj.replace_target_iter)

        rng_state = data.get("rng_state")
        if rng_state is not None:
            obj.rng.set_state(rng_state)

        return obj

