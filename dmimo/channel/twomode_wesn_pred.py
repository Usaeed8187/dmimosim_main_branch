import copy
import numpy as np
import tensorflow as tf

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation

class twomode_wesn_pred:

    def __init__(self, rc_config, num_freq_re, num_rx_ant, num_tx_ant, type=np.complex64):
        
        self.rc_config = rc_config
        self.dtype = type

        self.num_freq_re = num_freq_re
        self.N_r = num_rx_ant
        self.N_t = num_tx_ant

        self.sparsity = rc_config.W_tran_sparsity
        self.spectral_radius = rc_config.W_tran_radius
        self.input_scale = rc_config.input_scale
        self.window_length = rc_config.window_length
        self.reg = rc_config.regularization
        self.enable_window = rc_config.enable_window
        self.history_len = rc_config.history_len

        seed = 10
        self.RS = np.random.RandomState(seed)

        self.N_in_left = self.N_r
        if self.enable_window:
            self.N_in_right = self.N_t * self.window_length # TODO: only windowing on the transmit antenna axis for now. evaluate windowing on the receive antenna axis later
        else:
            self.N_in_right = self.N_t

        self.d_left = self.N_in_left # TODO: currently just basing on the size of the input. try other configurations
        self.d_right = self.N_in_right

        if self.d_left is None:
            self.d_left = self.N_r
        if self.d_right is None:
            self.d_right = self.N_t        

        self.init_weights()

    def init_weights(self):

        self.W_res_left = self.sparse_mat(self.d_left)
        self.W_res_right = self.sparse_mat(self.d_right)

        self.W_in_left = 2 * (self.RS.rand(self.d_left, self.N_in_left) - 0.5) # TODO: check if I should make this complex later
        self.W_in_right = 2 * (self.RS.rand(self.N_in_right, self.d_right) - 0.5) # TODO: check if I should make this complex later

        # TODO: using a vectorization trick to learn one vectorized W_out instead of left and right W_outs.
        # This is mathematically equivalent to 
        # self.W_out_left = self.RS.randn(self.N_r, self.d_left)
        # self.W_out_right = self.RS.randn(self.d_right + self.N_in_right, self.N_t)
        self.feature_dim = int(self.d_left * self.d_right * (self.window_length + 1))
        self.W_out = self.RS.randn(self.N_r * self.N_t, self.feature_dim).astype(self.dtype)        

        self.S_0 = np.zeros([self.d_left, self.d_right], dtype=self.dtype)

    
    def predict(self, h_freq_csi_history):

        h_freq_csi_predicted = self.pred_v2(h_freq_csi_history)

        return h_freq_csi_predicted
    
    def pred_v1(self, h_freq_csi_history):

        # v1 loops over freq REs and does a prediction for each freq RE independently. 
        # TODO: extend to threemode ESN to do everything at once
        # 
        # intended input size: 
        # [num_batches (time steps), 1, num_rx_nodes, num_rx_antennas, num_tx_nodes, num_tx_antennas, numm_ofdm_syms, num_freq_res (subcarriers or RBs)]
        # 
        # can be used per tx-rx pair if needed (eg. if the matrix sizes are too large). 
        # this should be handled through input dimensions. 
        # if num_rx_nodes = num_tx_nodes = 1, this function handles all nodes together. 
        # currently only supports nodes with the same amount of antennas 
        # (i.e. have to treat gNB as 2 UEs before passing to this function)
        # TODO: add support for heterogenous antenna sizes later
        
        if tf.rank(h_freq_csi_history).numpy() == 8:
            h_freq_csi_history = np.asarray(h_freq_csi_history).transpose([0,1,2,3,4,5,7,6])
            num_batches = h_freq_csi_history.shape[1]
            num_rx_nodes = h_freq_csi_history.shape[2]
            num_rx_antennas = h_freq_csi_history.shape[3]
            num_tx_nodes = h_freq_csi_history.shape[4]
            num_tx_antennas = h_freq_csi_history.shape[5]
            num_freq_res = h_freq_csi_history.shape[6]
            num_ofdm_syms = h_freq_csi_history.shape[7]
        else:
            raise ValueError("\n The dimensions of h_freq_csi_history are not correct")

        channel_train_input = h_freq_csi_history[:-1, ...]
        channel_train_gt = h_freq_csi_history[1:, ...]

        if not self.enable_window: # TODO: Test window weights later
            window_weights = None

        chan_pred = np.zeros(h_freq_csi_history[0,...].shape, dtype=self.dtype)
        for rx_node in range(num_rx_nodes):
            for tx_node in range(num_tx_nodes):
                for freq_re in range(num_freq_res):
                    for ofdm_sym in range(num_ofdm_syms):

                        self.init_weights()
                    
                        channel_train_input_temp = channel_train_input[:, 0, tx_node, :, rx_node, :, freq_re, ofdm_sym]

                        channel_train_gt_temp = channel_train_gt[:, 0, tx_node, :, rx_node, :, freq_re, ofdm_sym]

                        curr_train = self.fitting_time(channel_train_input_temp, channel_train_gt_temp, curr_window_weights=None)

                        channel_test_input = channel_train_gt_temp
                        channel_pred_temp = self.test_train_predict(channel_test_input, curr_window_weights=None)
                        channel_pred_temp = channel_pred_temp[:,:,-1:]
                        channel_pred_temp = np.squeeze(channel_pred_temp)
                        chan_pred[:, tx_node, :, rx_node, :, freq_re, ofdm_sym] = channel_pred_temp

        chan_pred = chan_pred.transpose([0,1,2,3,4,6,5])
        chan_pred = tf.convert_to_tensor(chan_pred)

        return chan_pred

    def pred_v2(self, h_freq_csi_history):
        
        
        if tf.rank(h_freq_csi_history).numpy() == 8:
            h_freq_csi_history = np.asarray(h_freq_csi_history).transpose([0,1,2,3,4,5,7,6])
            num_batches = h_freq_csi_history.shape[1]
            num_rx_nodes = h_freq_csi_history.shape[2]
            num_rx_antennas = h_freq_csi_history.shape[3]
            num_tx_nodes = h_freq_csi_history.shape[4]
            num_tx_antennas = h_freq_csi_history.shape[5]
            num_freq_res = h_freq_csi_history.shape[6]
            num_ofdm_syms = h_freq_csi_history.shape[7]
        else:
            raise ValueError("\n The dimensions of h_freq_csi_history are not correct")

        channel_train_input = h_freq_csi_history[:-1, ...]
        channel_train_gt    = h_freq_csi_history[1:,  ...]
        
        if not self.enable_window:
            window_weights = None

        chan_pred = np.zeros(h_freq_csi_history[0,...].shape, dtype=self.dtype)

        # === ONE reservoir per (rx_node, tx_node) pair; shared across all RBs ===
        for rx_node in range(num_rx_nodes):
            for tx_node in range(num_tx_nodes):

                # Initialize weights ONCE for all RBs of this (rx_node, tx_node)
                self.init_weights()

                # --------- (A) FEATURE BUILD PHASE: stack all RBs (and OFDM syms) ----------
                S_list, Y_list = [], []
                for freq_re in range(num_freq_res):
                    for ofdm_sym in range(num_ofdm_syms):
                        # Train sequences for this RB/symbol → [T, N_r, N_t]
                        Y_in  = channel_train_input[:, 0, tx_node, :, rx_node, :, freq_re, ofdm_sym]
                        Y_out = channel_train_gt[:,    0, tx_node, :, rx_node, :, freq_re, ofdm_sym]

                        # Optional: do NOT reset S_0 here if you want cross-RB continuity
                        # self.S_0 = np.zeros([self.d_left, self.d_right], dtype=self.dtype)

                        S_f, Y_f = self.build_S_Y(Y_in, Y_out, curr_window_weights=None)
                        S_list.append(S_f); Y_list.append(Y_f)

                S_all = np.concatenate(S_list, axis=1)  # (F, sum_T)
                Y_all = np.concatenate(Y_list, axis=1)  # (N_r*N_t, sum_T)

                # --------- (B) SINGLE READOUT SOLVE (shared across RBs) ----------
                # Prefer ridge for stability:
                G = self.reg_p_inv(S_all)               # (sum_T, F)  :=  S_all^H (S_all S_all^H + λI)^{-1}
                self.W_out = Y_all @ G                  # (N_r*N_t, F)

                # --------- (C) PREDICTION PHASE with the shared W_out ----------
                for freq_re in range(num_freq_res):
                    for ofdm_sym in range(num_ofdm_syms):
                        # Use last known channel as test input; predict next step
                        channel_test_input = channel_train_gt[:, 0, tx_node, :, rx_node, :, freq_re, ofdm_sym]

                        # Optional: either carry S_0 across RBs for smoothness,
                        # or reset it per RB. Start with reset; then try carry-over.
                        self.S_0 = np.zeros([self.d_left, self.d_right], dtype=self.dtype)

                        channel_pred_temp = self.test_train_predict(channel_test_input, curr_window_weights=None)
                        channel_pred_temp = channel_pred_temp[:, :, -1:]       # keep last step
                        channel_pred_temp = np.squeeze(channel_pred_temp)      # [N_r, N_t]
                        chan_pred[:, tx_node, :, rx_node, :, freq_re, ofdm_sym] = channel_pred_temp

        chan_pred = chan_pred.transpose([0,1,2,3,4,6,5])
        chan_pred = tf.convert_to_tensor(chan_pred)
        return chan_pred

    def build_S_Y(self, channel_input, channel_output, curr_window_weights):
        # channel_input, channel_output: [T, N_r, N_t]
        Y_3D = channel_input
        Y_target_3D = channel_output

        if self.enable_window:
            Y_3D_win = self.form_window_input_signal(Y_3D, curr_window_weights)
        else:
            # Safe fallback if forget_length not set:
            forget = getattr(self, "forget_length", 0)
            Y_3D_win = np.concatenate([Y_3D, np.zeros([Y_3D.shape[0], forget, Y_3D.shape[2]], dtype=self.dtype)], axis=1)

        S_3D_transit = self.state_transit(Y_3D_win * self.input_scale)
        S_3D = np.concatenate([S_3D_transit, Y_3D_win], axis=-1)

        T = S_3D.shape[0]
        S = np.column_stack([S_3D[t].reshape(-1, order='C') for t in range(T)])  # (feature_dim, T)
        Y = np.column_stack([Y_target_3D[t].reshape(-1, order='C') for t in range(T)])  # (N_r*N_t, T)
        return S, Y


    def calculate_window_weights(self, h_freq_csi_history):

        if self.window_weighting_method == 'autocorrelation':
            def autocorrelation(x):
                """Compute the autocorrelation of a 1D signal."""
                n = len(x)
                x_mean = np.mean(x)
                x_var = np.var(x)
                acf = np.correlate(x - x_mean, x - x_mean, mode='full') / (n * x_var)
                return acf[n-1:]  # Keep only non-negative lags

            h_reshaped = np.moveaxis(h_freq_csi_history, -1, 0)
            acf_result = np.apply_along_axis(autocorrelation, 0, h_reshaped)
            acf_result = np.squeeze(np.mean(acf_result, axis=-1))

            window_weights = np.abs(acf_result)
        elif self.window_weighting_method == 'same_weights':
            window_weights = 1
        elif self.window_weighting_method == 'exponential_decay':
            # x = np.linspace(0, self.window_length-1, self.history_len*self.num_ofdm_sym)
            x = np.linspace(0, self.window_length-1, h_freq_csi_history.shape[1])
            window_weights = np.exp(-x/2)
        elif self.window_weighting_method == 'none':
            window_weights = np.ones(h_freq_csi_history.shape[1])
        else:
            raise ValueError("\n The window_weighting_method specified is not implemented")
        
        return window_weights

    def sparse_mat(self, m):
        
        W = 2*(self.RS.rand(m, m) - 0.5) + 2j*(self.RS.rand(m, m) - 0.5)
        W[self.RS.rand(*W.shape) < self.sparsity] = 0+1j*0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W = W * (self.spectral_radius / radius)
        
        return W

    def complex_to_real_target(self, Y_target_2D):
        Y_target_2D_real_list = []
        for t in range(self.N_t):
            target = Y_target_2D[t, :].reshape(1, -1) # (1, N_symbols * (N_fft+N_cp))
            real_target = np.concatenate((np.real(target), np.imag(target)), axis=0)  # (2, N_symbols * (N_fft+N_cp))
            Y_target_2D_real_list.append(real_target)
        Y_target_2D_real = np.concatenate(Y_target_2D_real_list, axis=0)
        return Y_target_2D_real

    def fitting_time(self, channel_input, channel_output, curr_window_weights):

        Y_3D = channel_input
        Y_target_3D = channel_output

        if self.enable_window:
            Y_3D_new = self.form_window_input_signal(Y_3D, curr_window_weights)
        else:
            # TODO: not sure if this still works. add in forget length functionality later
            Y_3D_new = np.concatenate([Y_3D, np.zeros([Y_3D.shape[0], self.forget_length], dtype=self.dtype)], axis=1)

        S_3D_transit = self.state_transit(Y_3D_new * self.input_scale)

        S_3D = np.concatenate([S_3D_transit, Y_3D_new], axis=-1)

        # vectorization trick. equivalent to having two W_out matrices on either side of the feature matrix being fed to the output
        T = S_3D.shape[0]
        S = np.column_stack([
            S_3D[t].reshape(-1, order='C') for t in range(T)
        ])  # (feature_dim, T)

        Y = np.column_stack([
            Y_target_3D[t].reshape(-1, order='C') for t in range(T)
        ])
        
        self.W_out = Y @ np.linalg.pinv(S)
        
        pred_channel = self.W_out @ S

        pred_channel = pred_channel.reshape([self.N_r, self.N_t, -1])

        return pred_channel

    def cal_nmse(self, H, H_hat):
        H_hat = tf.cast(H_hat, dtype=H.dtype)
        mse = np.sum(np.abs(H - H_hat) ** 2)
        normalization_factor = np.sum((np.abs(H) + np.abs(H_hat)) ** 2)
        nmse = mse / normalization_factor
        return nmse

    def reg_p_inv(self, X):
        # X: (F, T)
        F = X.shape[0]
        G = X @ X.conj().T + self.reg * np.eye(F, dtype=self.dtype)  # (F,F)
        G = X.conj().T @ np.linalg.pinv(G)                 # (T,F)

        return G

    def form_window_input_signal(self, Y_3D_complex, curr_window_weights):

        # if self.window_weight_application == 'across_inputs' or self.window_weight_application == 'across_time_and_inputs':
        #     Y_2D_complex = Y_2D_complex * curr_window_weights

        assert Y_3D_complex.ndim == 3, "Y must be [T, N_r, N_t]"
        T, N_r, N_t = Y_3D_complex.shape
        L = int(self.window_length)

        Y_3D_window = np.zeros((T, N_r, L * N_t), dtype=self.dtype)

        for k in range(T):
            blocks = []
            for ell in range(L):
                t = k - ell
                if t >= 0:
                    blocks.append(Y_3D_complex[t])       # [N_r, N_t]
                else:
                    blocks.append(np.zeros((N_r, N_t), dtype=self.dtype))  # causal zero-pad
            # Concatenate along Tx axis → [N_r, L*N_t]
            Y_3D_window[k] = np.concatenate(blocks, axis=-1)

        return Y_3D_window

    def test_train_predict(self, channel_train_input, curr_window_weights):
        self.S_0 = np.zeros([self.d_left, self.d_right], dtype=self.dtype)

        Y_3D_org = channel_train_input

        Y_3D = self.form_window_input_signal(Y_3D_org, curr_window_weights)

        S_3D = self.state_transit(Y_3D * self.input_scale)

        S_3D = np.concatenate([S_3D, Y_3D], axis=-1)

        # vectorization trick. equivalent to having two W_out matrices on either side of the feature matrix being fed to the output
        T = S_3D.shape[0]
        S = np.column_stack([
            S_3D[t].reshape(-1, order='C') for t in range(T)
        ])  # (feature_dim, T)

        curr_channel_pred = self.W_out @ S

        curr_channel_pred = curr_channel_pred.reshape([self.N_r, self.N_t, -1])

        return curr_channel_pred

    def state_transit(self, Y_3D):

        T = Y_3D.shape[0] # number of samples

        S_2D = copy.deepcopy(self.S_0)
        S_3D = []
        for t in range(T):
            S_2D = self.complex_tanh(self.W_res_left @ S_2D @ self.W_res_right + self.W_in_left @ Y_3D[t,:,:] @ self.W_in_right)
            S_3D.append(S_2D)

        S_3D = np.stack(S_3D, axis=0)

        self.S_0 = S_2D

        return S_3D

    def complex_tanh(self, Y):
        return np.tanh(np.real(Y)) + 1j * np.tanh(np.imag(Y))