# dMIMO network scenarios

from .config import Config


class RCConfig(Config):

    def __init__(self, **kwargs):
        self._name = "RC Configuration"
        self._num_neurons = 16  # 256, 16, 8
        self._W_tran_sparsity = 0.4  # 0.1, 0.4
        self._W_tran_radius = 0.5
        self._input_scale = 0.8 
        self._initial_forget_length = 0
        self._max_forget_length = 1
        self._forget_length_search_step = 1
        self._window_length = 1
        self._learning_delay = True
        self._enable_window = False
        self._regularization = 1
        self._type = 'complex' # real, complex
        self._DF_rls = False
        self._history_len = 8 # number of subframes that we use to train
        self._prediction_on = True
        self._treatment = 'SISO' # SISO, vectorized_MIMO, piece_wise_vectorized_MIMO, TODO: piece_wise_MIMO not currently implemented


        super().__init__(**kwargs)

    @property
    def num_neurons(self):
        return self._num_neurons

    @num_neurons.setter
    def num_neurons(self, val):
        self._num_neurons = val

    @property
    def W_tran_sparsity(self):
        return self._W_tran_sparsity

    @W_tran_sparsity.setter
    def W_tran_sparsity(self, val):
        self._W_tran_sparsity = val

    @property
    def W_tran_radius(self):
        return self._W_tran_radius

    @W_tran_radius.setter
    def W_tran_radius(self, val):
        self._W_tran_radius = val

    @property
    def input_scale(self):
        return self._input_scale

    @input_scale.setter
    def input_scale(self, val):
        self._input_scale = val

    @property
    def initial_forget_length(self):
        return self._initial_forget_length

    @initial_forget_length.setter
    def initial_forget_length(self, val):
        self._initial_forget_length = val

    @property
    def max_forget_length(self):
        return self._max_forget_length

    @max_forget_length.setter
    def max_forget_length(self, val):
        self._max_forget_length = val

    @property
    def forget_length_search_step(self):
        return self._forget_length_search_step

    @forget_length_search_step.setter
    def forget_length_search_step(self, val):
        self._forget_length_search_step = val

    @property
    def window_length(self):
        return self._window_length

    @window_length.setter
    def window_length(self, val):
        self._window_length = val

    @property
    def learning_delay(self):
        return self._learning_delay

    @learning_delay.setter
    def learning_delay(self, val):
        self._learning_delay = val

    @property
    def enable_window(self):
        return self._enable_window

    @enable_window.setter
    def enable_window(self, val):
        self._enable_window = val

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, val):
        self._regularization = val

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, val):
        self._type = val

    @property
    def DF_rls(self):
        return self._DF_rls

    @DF_rls.setter
    def DF_rls(self, val):
        self._DF_rls = val

    @property
    def history_len(self):
        return self._history_len

    @history_len.setter
    def history_len(self, val):
        self._history_len = val
    
    @property
    def prediction_on(self):
        return self._prediction_on

    @prediction_on.setter
    def prediction_on(self, val):
        self._prediction_on = val
    
    @property
    def treatment(self):
        return self._treatment

    @treatment.setter
    def treatment(self, val):
        self._treatment = val
