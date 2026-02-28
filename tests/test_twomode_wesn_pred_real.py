import copy
import numpy as np
import tensorflow as tf

class twomode_wesn_pred:

    def __init__(self, history_length=9, num_rx_ant=2, num_tx_ant=2, type=np.float32):

        self.dtype = type

        self.N_r = num_rx_ant
        self.N_t = num_tx_ant

        self.sparsity = 0.4
        self.spectral_radius = 0.5
        self.input_scale = 0.8
        self.window_length = 3
        self.reg = 1
        self.enable_window = True
        self.history_len = history_length

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

        if np.isnan(self.W_res_left).any() or np.isnan(self.W_res_right).any():
            raise ValueError("\n NaN values found in reservoir weights")

        self.W_in_left = 2 * (self.RS.rand(self.d_left, self.N_in_left) - 0.5)
        self.W_in_right = 2 * (self.RS.rand(self.N_in_right, self.d_right) - 0.5)

        # TODO: using a vectorization trick to learn one vectorized W_out instead of left and right W_outs.
        # This is mathematically equivalent to 
        # self.W_out_left = self.RS.randn(self.N_r, self.d_left)
        # self.W_out_right = self.RS.randn(self.d_right + self.N_in_right, self.N_t)
        self.feature_dim = int(self.d_left * self.d_right * (self.window_length + 1))
        self.W_out = self.RS.randn(self.N_r * self.N_t, self.feature_dim).astype(self.dtype)        

        self.S_0 = np.zeros([self.d_left, self.d_right], dtype=self.dtype)

    
    def predict(self, h_freq_csi_history):

        h_freq_csi_predicted = self.pred(h_freq_csi_history)

        return h_freq_csi_predicted
    
    def pred(self, h_freq_csi_history):
        
        h_freq_csi_history = np.asarray(h_freq_csi_history)

        T, RxAnt, TxAnt, F = h_freq_csi_history.shape

        channel_train_input = h_freq_csi_history[:-1, ...]
        channel_train_gt    = h_freq_csi_history[1:,  ...]
        
        chan_pred = np.zeros(h_freq_csi_history[0,...].shape, dtype=self.dtype)

        # === ONE reservoir shared across all RBs ===

        # Initialize weights ONCE for all RBs of this (rx_node, tx_node)
        self.init_weights()

        # --------- (A) FEATURE BUILD PHASE: stack all RBs (and OFDM syms) ----------
        S_list, Y_list = [], []
        for freq_re in range(F):
            # Train sequences for this RB/symbol → [T, N_r, N_t]
            Y_in  = channel_train_input[:, :, :, freq_re]
            Y_out = channel_train_gt[:, :, :, freq_re]

            S_f, Y_f = self.build_S_Y(Y_in, Y_out, curr_window_weights=None)
            S_list.append(S_f); Y_list.append(Y_f)
                
        S_all = np.concatenate(S_list, axis=1)  # (F, sum_T)
        Y_all = np.concatenate(Y_list, axis=1)  # (N_r*N_t, sum_T)

        # --------- (B) SINGLE READOUT SOLVE (shared across RBs) ----------
        # Prefer ridge for stability:
        G = self.reg_p_inv(S_all)               # (sum_T, F)  :=  S_all^H (S_all S_all^H + λI)^{-1}
        self.W_out = Y_all @ G                  # (N_r*N_t, F)

        # --------- (C) PREDICTION PHASE with the shared W_out ----------
        for freq_re in range(F):
            # Use last known channel as test input; predict next step
            channel_test_input = channel_train_gt[:, :, :, freq_re]

            # Optional: either carry S_0 across RBs for smoothness,
            # or reset it per RB. Start with reset; then try carry-over.
            self.S_0 = np.zeros([self.d_left, self.d_right], dtype=self.dtype)

            channel_pred_temp = self.test_train_predict(channel_test_input, curr_window_weights=None)
            channel_pred_temp = channel_pred_temp[:, :, -1:]       # keep last step
            channel_pred_temp = np.squeeze(channel_pred_temp)      # [N_r, N_t]
            chan_pred[:, :, freq_re] = channel_pred_temp

        return chan_pred

    def build_S_Y(self, channel_input, channel_output, curr_window_weights):
        # channel_input, channel_output: [T, N_r, N_t]
        Y_3D = channel_input
        Y_target_3D = channel_output

        if np.isnan(channel_input).any():
            hold = 1

        if np.isnan(channel_output).any():
            hold = 1

        if self.enable_window:
            Y_3D_win = self.form_window_input_signal(Y_3D, curr_window_weights)
        else:
            # Safe fallback if forget_length not set:
            forget = getattr(self, "forget_length", 0)
            Y_3D_win = np.concatenate([Y_3D, np.zeros([Y_3D.shape[0], forget, Y_3D.shape[2]], dtype=self.dtype)], axis=1)
        
        if np.isnan(Y_3D_win).any():
            hold = 1

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

    def sparse_mat(self, m, max_tries=100, eps=1e-12):
        for attempt in range(max_tries):
            W = 2 * (self.RS.rand(m, m) - 0.5)
            W[self.RS.rand(m, m) < self.sparsity] = 0

            # Reject all-zero matrix immediately
            if not np.any(W):
                continue

            radius = np.max(np.abs(np.linalg.eigvals(W)))

            if np.isfinite(radius) and radius > eps:
                return W * (self.spectral_radius / radius)

        raise RuntimeError(
            f"Failed to generate a valid sparse matrix after {max_tries} attempts. "
            f"m={m}, sparsity={self.sparsity}, eps={eps}. "
            "Spectral radius was zero or numerically unstable each time."
        )

    def ensure_real_target(self, Y_target_2D):
        return np.asarray(Y_target_2D, dtype=self.dtype)

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
        G = X @ X.T + self.reg * np.eye(F, dtype=self.dtype)  # (F,F)
        G = X.T @ np.linalg.pinv(G)                 # (T,F)

        return G

    def form_window_input_signal(self, Y_3D_real, curr_window_weights):

        assert Y_3D_real.ndim == 3, "Y must be [T, N_r, N_t]"
        T, N_r, N_t = Y_3D_real.shape
        L = int(self.window_length)

        Y_3D_window = np.zeros((T, N_r, L * N_t), dtype=self.dtype)

        for k in range(T):
            blocks = []
            for ell in range(L):
                t = k - ell
                if t >= 0:
                    blocks.append(Y_3D_real[t])       # [N_r, N_t]
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
            S_2D = self.real_tanh(self.W_res_left @ S_2D @ self.W_res_right + self.W_in_left @ Y_3D[t,:,:] @ self.W_in_right)

            if np.isnan(S_2D).any():
                hold = 1
            S_3D.append(S_2D)

        S_3D = np.stack(S_3D, axis=0)

        self.S_0 = S_2D

        return S_3D

    def real_tanh(self, Y):
        return np.tanh(Y)

def _predict_pair_worker(args):
    base_history, rc_config, RB, tx_ant_idx, rx_ant_idx = args

    curr_h_freq_csi_history = base_history[:, :, :, rx_ant_idx, :, ...]
    curr_h_freq_csi_history = curr_h_freq_csi_history[:, :, :, :, :, tx_ant_idx, ...]

    twomode_predictor = twomode_wesn_pred(
        rc_config=rc_config,
        num_freq_re=RB,
        num_rx_ant=len(rx_ant_idx),
        num_tx_ant=len(tx_ant_idx),
    )

    tmp = twomode_predictor.predict(curr_h_freq_csi_history)
    rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
    return rx_idx, tx_idx, tmp
    

def predict_real(h_freq_csi_history):

    T, RxAnt, TxAnt, F = h_freq_csi_history.shape
    h_freq_csi = np.zeros(h_freq_csi_history[0, ...].shape, dtype=h_freq_csi_history.dtype)
    h_freq_csi_history = np.asarray(h_freq_csi_history)

    twomode_predictor = twomode_wesn_pred(h_freq_csi_history)
    h_freq_csi = np.asarray(twomode_predictor.predict(h_freq_csi_history))

    return h_freq_csi