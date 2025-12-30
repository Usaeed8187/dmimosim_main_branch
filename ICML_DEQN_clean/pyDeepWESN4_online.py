import numpy as np
import copy


def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.

    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s

    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


class WESN():

    # Modification needed for WESN
    def __init__(self, n_inputs, n_outputs, n_reservoir, n_layers, memory_size,
                 input_window_length, output_window_length,
                 spectral_radius=0.95, sparsity=0, noise=0.001, lr=0.01,
                 input_shift=None, input_scaling=None,
                 teacher_scaling=None, teacher_shift=None, random_seed=1,
                 out_activation=lambda x: x, inverse_out_activation=lambda x: x,
                 random_state=10):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            n_layers: nr of layers
            memory_size: how many time steps to keep in memory for training
            input_window_length: The number of past inputs used for the input window
            output_window_length: The number of past inputs used for the output window
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
        """
        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.memory_size = memory_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_window_length = input_window_length
        self.output_window_length = output_window_length
        self.input_window = np.zeros([input_window_length,n_inputs])
        self.output_window = np.zeros([output_window_length,n_outputs])
        
        # Used for giving a fixed bias or scaling to the input. It should not be used, it is not complete.
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state

        self.lr = lr
        self.W_out_type = 0

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self._initweights(random_seed)

        #self.extended_states = np.zeros((self.memory_size, self.n_reservoir * self.n_layers + int(self.n_inputs / 2)))
        if self.W_out_type == 1:
            self.extended_states = np.zeros((self.memory_size, self.n_reservoir * self.n_layers + self.n_inputs))
        else:
            # self.extended_states = np.zeros((self.memory_size, self.n_reservoir * self.n_layers))
            self.extended_states = np.zeros((self.memory_size, (self.output_window_length)*self.n_inputs +  self.n_layers * self.n_reservoir))
        self.memory_counter = 0

        self.laststate = []
        for i in range(n_layers):
            self.laststate.append(np.zeros(self.n_reservoir))
        # self.lastinput = np.zeros(self.n_inputs)
        self.lastinput = np.zeros([self.input_window_length,self.n_inputs])
        self.lastoutput = np.zeros(self.n_outputs)

        self.startstate = []
        for i in range(n_layers):
            self.startstate.append(np.zeros(self.n_reservoir))
        self.startinput = np.zeros(self.n_inputs)
        self.startoutput = np.zeros(self.n_outputs)

    # Modification needed for WESN
    def _initweights(self, random_seed):
        RandomState = np.random.RandomState(seed=random_seed)
        # np.random.seed(random_seed)
        # initialize recurrent weights:
        self.W = []
        for i in range(self.n_layers):
            # begin with a random matrix centered around zero:
            W_tmp = RandomState.uniform(-0.5, 0.5, size=(self.n_reservoir, self.n_reservoir))
            # delete the fraction of connections given by (self.sparsity):
            W_tmp[RandomState.uniform(0, 1, size=(W_tmp.shape[0], W_tmp.shape[1])) < self.sparsity] = 0
            # compute the spectral radius of these weights:
            # radius = np.max(np.abs(np.linalg.eigvals(W_tmp)))
            radius = np.max(np.abs(np.linalg.norm(W_tmp, 2)))
            # rescale them to reach the requested spectral radius:
            self.W.append(W_tmp * (self.spectral_radius / radius))

        # random input weights:
        self.W_in = []
        for i in range(self.n_layers):
            if (i == 0):
                #self.W_in.append(RandomState.uniform(-1, 1, size=(self.n_reservoir, int(self.n_inputs/2))))
                self.W_in.append(RandomState.uniform(-1, 1, size=(self.n_reservoir, self.n_inputs*(self.input_window_length+1))))
            else:
                self.W_in.append(RandomState.uniform(-1, 1, size=(self.n_reservoir, self.n_reservoir)))

            r = np.max(np.abs(np.linalg.norm(self.W_in[i], 2)))
            self.W_in[i] = self.W_in[i] * (1 / r) # To make sure the spectral radius of the input weight matrix is 1
            # self.W_in[i] = self.W_in[i] * (self.spectral_radius / r) # This line was added by Ramin

        # random output weights:
        #self.W_out = RandomState.uniform(-1, 1, size=(self.n_outputs, self.n_reservoir * self.n_layers + int(self.n_inputs/2)))
        if self.W_out_type == 1:
            self.W_out = RandomState.uniform(-1, 1, size=(self.n_outputs, self.n_reservoir * self.n_layers + self.n_inputs))
        else:
            self.W_out = RandomState.uniform(-1, 1, size=(self.n_outputs, self.n_reservoir * self.n_layers + self.output_window_length*self.n_inputs))
        r = np.max(np.abs(np.linalg.norm(self.W_out, 2)))
        self.W_out = self.W_out * (1 / r) # To make the spectral radius of output weight matrix to be zero

    # Modification probably needed for WESN
    def _update(self, state:np.ndarray, input:np.ndarray, a=1):
        """performs one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        assert len(state) == self.n_layers
        for layer_state in state: assert layer_state.shape == (self.n_reservoir,) , f'The state\'s shape should be {(self.n_reservoir,)} but it is {layer_state.shape}'
        assert input.shape == ((self.input_window_length+1) * (self.n_inputs), ),\
        f'input must be of shape {((self.input_window_length+1) * (self.n_inputs), )}, but it is of shape {input.shape}'
        # I thought the shape of input should be ((self.input_window_length+1) * (self.n_inputs) + self.n_reservoir, ), but I had mistaken
        # with what the extended_states shape should be (which is still wrong BTW)
        state_new = []
        #tmp_input = copy.deepcopy(input[: int(self.n_inputs/2)])
        tmp_input = copy.deepcopy(input)
        # tmp_input = np.concatenate(self.input_window, np.reshape(input,[1,self.n_inputs]), axis=0)
        # self.input_window = tmp_input[1:,:] # An update to the input window
        for i in range(self.n_layers):
            state_new.append((1-a) * state[i] +
                             a * np.tanh(np.dot(self.W[i], state[i]) + np.dot(self.W_in[i], tmp_input)))
            tmp_input = state_new[i]

        return state_new

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    # Modification needed for WESN (actually, maybe not!)
    def fit(self, outputs, index, gradient_clip=10): # gradient_clip=10
        """
        Collect the network's reaction to training data, train readout weights.

        Args:
            inputs: array of dimensions (N_training_samples x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states

        Returns:
            the network's output on the training data, using the trained weights
        """
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        teachers_scaled = self._scale_teacher(outputs)

        # train the output weights
        teachers_scaled_train = teachers_scaled

        states_train = self.extended_states[index, :]
        error_train = np.dot(self.W_out, states_train.T) - teachers_scaled_train.T

        gradient = np.dot(error_train, states_train)
        #print(gradient / error_train.shape[1])
        sample_count = error_train.shape[1]
        if sample_count == 0:
            return

        gradient[gradient / error_train.shape[1] > gradient_clip] = gradient_clip * error_train.shape[1]
        gradient[gradient / error_train.shape[1] < -gradient_clip] = -gradient_clip * error_train.shape[1]
        self.W_out = self.W_out - self.lr * gradient / error_train.shape[1]

    # Modification needed for WESN
    def predict(self, inputs, nForgetPoints, continuation=True):
        """
        Apply the learned weights to the network's reactions to new input.

        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state

        Returns:
            Array of output activations
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = self.startstate
            lastinput = self.startinput
            lastoutput = self.startoutput
            self.input_window = np.zeros(self.input_window_length,self.n_inputs)

        # inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        inputs = np.vstack([self.input_window , self._scale_inputs(inputs)])
        states = []
        states.append(laststate)
        outputs = np.vstack(
            [lastoutput, np.zeros((n_samples, self.n_outputs))])

        for n in range(n_samples):
            # states.append(self._update(states[n], inputs[n + 1, :]))
            states.append(self._update(states[n], inputs[n : n + self.input_window_length + 1, :].flatten()))
            if self.W_out_type == 1:
                self.extended_states[self.memory_counter, : self.n_inputs] = inputs[n + 1, :]
            else:
                # Fill the first few elements of self.extended_states with the inputs. self.extended_states includes the 
                # arrays that are fed to the output weight matrix. These are supposed to be used later for training.
                n_out_window_elements = (self.output_window_length) * self.n_inputs
                self.extended_states[self.memory_counter, : n_out_window_elements] = inputs[n + 1 : n + self.output_window_length + 1, :].flatten()
            for i in range(self.n_layers):
                '''
                self.extended_states[self.memory_counter,
                int(self.n_inputs/2) + i * self.n_reservoir :
                int(self.n_inputs/2) + (i+1) * self.n_reservoir] = states[n+1][i]
                '''
                if self.W_out_type == 1:
                    self.extended_states[self.memory_counter,
                    self.n_inputs + i * self.n_reservoir:
                    self.n_inputs + (i + 1) * self.n_reservoir] = states[n + 1][i]
                else:
                    self.extended_states[self.memory_counter,
                    n_out_window_elements + i * self.n_reservoir:
                    n_out_window_elements + (i + 1) * self.n_reservoir] = states[n + 1][i]
            outputs[n + 1, :] = self.out_activation(np.dot(self.W_out, self.extended_states[self.memory_counter, :]))
            self.memory_counter = self.memory_counter + 1


        if continuation:
            # remember the last state for later:
            self.laststate = states[-1]
            self.lastinput = inputs[-1,:]
            self.lastoutput = outputs[-1, :]
            self.input_window = inputs[-self.input_window_length:,:]

        transient = nForgetPoints + 1

        return self._unscale_teacher(self.out_activation(outputs[transient:]))

    # Modification needed for WESN
    def predict_training(self, index):
        """
        Apply the learned weights to the network's reactions to new input.

        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state

        Returns:
            Array of output activations
        """
        #extended_states = self.extended_states[index_start : index_start + training_batch_size, :]
        extended_states = self.extended_states[index, :]
        outputs = self.out_activation(np.dot(self.W_out, extended_states.T))
        outputs = outputs.T

        '''
        outputs = np.zeros((training_batch_size, self.n_outputs))
        for n in range(training_batch_size):
            outputs[n, :] = self.out_activation(np.dot(self.W_out, self.extended_states[index_start + n, :]))
        '''
        return self._unscale_teacher(self.out_activation(outputs))

    # Modification needed for WESN
    def refresh_state(self):
        # self.extended_states = np.zeros((self.memory_size, self.n_reservoir * self.n_layers + int(self.n_inputs/2)))
        if self.W_out_type == 1:
            self.extended_states = np.zeros((self.memory_size, self.n_reservoir * self.n_layers + self.n_inputs))
        else:
            self.extended_states = np.zeros((self.memory_size, (self.output_window_length)*self.n_inputs +  self.n_layers * self.n_reservoir))
        self.memory_counter = 0
        self.input_window = np.zeros([self.input_window_length, self.n_inputs])