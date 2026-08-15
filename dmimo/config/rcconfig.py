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
        self._enable_kalman_weight_config = False
        self._kalman_gain_iters = 100
        self._kalman_eps = 1e-8
        self._state_dim_left = None
        self._state_dim_right = None
        self._esn_m = 4
        self._esn_k = 4
        self._esn_num_freqs = 64
        self._esn_activation = 'tanh'
        self._esn_ls_reg = 1e-6
        self._esn_diagnostics = False
        self._enable_skip_connections = True
        self._enable_residue_low_rank = False
        self._residue_energy_threshold = 0.95
        self._reservoir_readout_regularization = 1e-2
        self._skip_readout_regularization = 1e-4
        self._wesn_lite_readout_mode = 'centered_ridge'
        self._wesn_lite_subcarriers_per_rb = 12
        self._enable_balanced_truncation = False
        self._enable_balanced_hankel_truncation = False
        self._balanced_hankel_energy_threshold = 0.90

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
    
    @property
    def enable_kalman_weight_config(self):
        return self._enable_kalman_weight_config

    @enable_kalman_weight_config.setter
    def enable_kalman_weight_config(self, val):
        self._enable_kalman_weight_config = val

    @property
    def kalman_gain_iters(self):
        return self._kalman_gain_iters

    @kalman_gain_iters.setter
    def kalman_gain_iters(self, val):
        self._kalman_gain_iters = val

    @property
    def kalman_eps(self):
        return self._kalman_eps

    @kalman_eps.setter
    def kalman_eps(self, val):
        self._kalman_eps = val

    @property
    def state_dim_setting(self):
        return self._state_dim_setting

    @state_dim_setting.setter
    def state_dim_setting(self, val):
        self._state_dim_setting = val

    @property
    def state_dim_left(self):
        return self._state_dim_left

    @state_dim_left.setter
    def state_dim_left(self, val):
        self._state_dim_left = val

    @property
    def state_dim_right(self):
        return self._state_dim_right

    @state_dim_right.setter
    def state_dim_right(self, val):
        self._state_dim_right = val

    @property
    def esn_m(self):
        return self._esn_m

    @esn_m.setter
    def esn_m(self, val):
        self._esn_m = val

    @property
    def esn_k(self):
        return self._esn_k

    @esn_k.setter
    def esn_k(self, val):
        self._esn_k = val

    @property
    def esn_num_freqs(self):
        return self._esn_num_freqs

    @esn_num_freqs.setter
    def esn_num_freqs(self, val):
        self._esn_num_freqs = val

    @property
    def esn_activation(self):
        return self._esn_activation

    @esn_activation.setter
    def esn_activation(self, val):
        self._esn_activation = val

    @property
    def esn_ls_reg(self):
        return self._esn_ls_reg

    @esn_ls_reg.setter
    def esn_ls_reg(self, val):
        self._esn_ls_reg = val

    @property
    def esn_diagnostics(self):
        return self._esn_diagnostics

    @esn_diagnostics.setter
    def esn_diagnostics(self, val):
        self._esn_diagnostics = val

    @property
    def enable_skip_connections(self):
        return self._enable_skip_connections

    @enable_skip_connections.setter
    def enable_skip_connections(self, val):
        self._enable_skip_connections = val

    @property
    def enable_residue_low_rank(self):
        return self._enable_residue_low_rank

    @enable_residue_low_rank.setter
    def enable_residue_low_rank(self, val):
        self._enable_residue_low_rank = val

    @property
    def residue_energy_threshold(self):
        return self._residue_energy_threshold

    @residue_energy_threshold.setter
    def residue_energy_threshold(self, val):
        self._residue_energy_threshold = val

    @property
    def reservoir_readout_regularization(self):
        return self._reservoir_readout_regularization

    @reservoir_readout_regularization.setter
    def reservoir_readout_regularization(self, val):
        self._reservoir_readout_regularization = val

    @property
    def skip_readout_regularization(self):
        return self._skip_readout_regularization

    @skip_readout_regularization.setter
    def skip_readout_regularization(self, val):
        self._skip_readout_regularization = val

    @property
    def wesn_lite_readout_mode(self):
        return self._wesn_lite_readout_mode

    @wesn_lite_readout_mode.setter
    def wesn_lite_readout_mode(self, val):
        self._wesn_lite_readout_mode = val

    @property
    def wesn_lite_subcarriers_per_rb(self):
        return self._wesn_lite_subcarriers_per_rb

    @wesn_lite_subcarriers_per_rb.setter
    def wesn_lite_subcarriers_per_rb(self, val):
        self._wesn_lite_subcarriers_per_rb = val

    @property
    def enable_balanced_truncation(self):
        return self._enable_balanced_truncation

    @enable_balanced_truncation.setter
    def enable_balanced_truncation(self, val):
        self._enable_balanced_truncation = bool(val)

    @property
    def enable_balanced_hankel_truncation(self):
        return self._enable_balanced_hankel_truncation

    @enable_balanced_hankel_truncation.setter
    def enable_balanced_hankel_truncation(self, val):
        self._enable_balanced_hankel_truncation = bool(val)

    @property
    def balanced_hankel_energy_threshold(self):
        return self._balanced_hankel_energy_threshold

    @balanced_hankel_energy_threshold.setter
    def balanced_hankel_energy_threshold(self, val):
        self._balanced_hankel_energy_threshold = float(val)
