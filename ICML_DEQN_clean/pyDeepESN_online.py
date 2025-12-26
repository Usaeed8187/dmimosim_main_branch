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


class ESN():

    def __init__(self, n_inputs, n_outputs, n_reservoir=200, n_layers=1,
                 spectral_radius=0.95, sparsity=0, noise=0.001, lr=0.01, teacher_forcing=False,
                 input_shift=None, input_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=lambda x: x, inverse_out_activation=lambda x: x,
                 random_state=None):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            n_layers: nr of layers
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
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise

        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state

        self.lr = lr

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

        self.teacher_forcing = teacher_forcing
        self.initweights()

        self.laststate = []
        for i in range(n_layers):
            self.laststate.append(np.zeros(self.n_reservoir))
        self.lastinput = np.zeros(self.n_inputs)
        self.lastoutput = np.zeros(self.n_outputs)

        self.startstate = []
        for i in range(n_layers):
            self.startstate.append(np.zeros(self.n_reservoir))
        self.startinput = np.zeros(self.n_inputs)
        self.startoutput = np.zeros(self.n_outputs)

    def initweights(self):
        # initialize recurrent weights:
        self.W = []
        for i in range(self.n_layers):
            # begin with a random matrix centered around zero:
            W_tmp = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
            # delete the fraction of connections given by (self.sparsity):
            W_tmp[self.random_state_.rand(W_tmp.shape[0], W_tmp.shape[1]) < self.sparsity] = 0
            # compute the spectral radius of these weights:
            radius = np.max(np.abs(np.linalg.eigvals(W_tmp)))
            # rescale them to reach the requested spectral radius:
            self.W.append(W_tmp * (self.spectral_radius / radius))

        # random input weights:
        self.W_in = []
        for i in range(self.n_layers):
            if (i==0):
                self.W_in.append(self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1)
            else:
                self.W_in.append(self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1)

        # random feedback (teacher forcing) weights:
        self.W_feedb = []
        for i in range(self.n_layers):
            self.W_feedb.append(self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1)

        # random output weights:
        self.W_out = []
        for i in range(self.n_layers):
            self.W_out.append(self.random_state_.rand(self.n_outputs, self.n_reservoir + self.n_inputs) * 2 - 1)

    def _update(self, state, input, output):
        """performs one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        state_new = []
        if self.teacher_forcing:
            tmp_input = copy.deepcopy(input)
            for i in range(self.n_layers):
                state_new.append(np.tanh(np.dot(self.W[i], state[i])
                                 + np.dot(self.W_in[i], tmp_input)
                                 + np.dot(self.W_feedb[i], output)))
                tmp_input = np.dot(self.W_out[i], np.hstack((state_new[i], input)))
        else:
            tmp_input = copy.deepcopy(input)
            for i in range(self.n_layers):
                state_new.append(np.tanh(np.dot(self.W[i], state[i])
                                         + np.dot(self.W_in[i], tmp_input)))
                tmp_input = np.dot(self.W_out[i], np.hstack((state_new[i], input)))

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

    def fit(self, inputs, outputs, nForgetPoints):
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
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        # step the reservoir through the given input,output pairs:
        states = []
        states.append(self.startstate)
        inputs_scaled = np.vstack(
            [self.startinput, inputs_scaled])
        teachers_scaled = np.vstack(
            [self.startoutput, teachers_scaled])

        for n in range(inputs.shape[0]):
            states.append(self._update(states[n], inputs_scaled[n + 1, :], teachers_scaled[n, :]))

        # train the output weights of each layer
        teachers_scaled_train = teachers_scaled[nForgetPoints + 1:, :]
        states_train_list = states[nForgetPoints + 1:]
        for i in range(self.n_layers):
            states_train = np.zeros((len(states_train_list), self.n_reservoir + self.n_inputs))
            states_train[:, self.n_reservoir:] = inputs_scaled[nForgetPoints + 1:, :]
            for n in range(len(states_train_list)):
                states_train[n, :self.n_reservoir] = states_train_list[n][i]
            error_train = np.dot(self.W_out[i], states_train.T) - teachers_scaled_train.T
            '''
            gamma = 0.9
            for k in range(error_train.shape[1]):
                error_train[:, k] = error_train[:, k] * gamma**(error_train.shape[1] - k - 1)
            '''
            gradient = np.dot(error_train, states_train)
            #self.W_out[i] = self.W_out[i] - self.lr * gradient / np.linalg.norm(gradient)
            self.W_out[i] = self.W_out[i] - self.lr * gradient / error_train.shape[1]

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

        inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        states = []
        states.append(laststate)
        outputs = np.vstack(
            [lastoutput, np.zeros((n_samples, self.n_outputs))])

        for n in range(n_samples):
            states.append(self._update(states[n], inputs[n + 1, :], outputs[n, :]))
            outputs[n + 1, :] = self.out_activation(np.dot(
                self.W_out[self.n_layers-1], np.hstack((states[n+1][self.n_layers-1], inputs[n + 1, :]))))


        if continuation:
            # remember the last state for later:
            self.laststate = states[-1]
            self.lastinput = inputs[-1, :]
            self.lastoutput = outputs[-1, :]

        transient = nForgetPoints + 1

        return self._unscale_teacher(self.out_activation(outputs[transient:]))