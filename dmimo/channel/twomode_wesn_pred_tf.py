import tensorflow as tf

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation

class twomode_wesn_pred_tf:

    def __init__(self, rc_config, num_freq_re, num_rx_ant, num_tx_ant, type=tf.complex64):
        
        self.rc_config = rc_config
        self.dtype = tf.as_dtype(type)
        self.real_dtype = self.dtype.real_dtype

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
        self.rng = tf.random.Generator.from_seed(seed)

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

    def random_uniform(self, shape):
        return self.rng.uniform(shape=shape, minval=-1.0, maxval=1.0, dtype=self.real_dtype)

    def init_weights(self):

        self.W_res_left = self.sparse_mat(self.d_left)
        self.W_res_right = self.sparse_mat(self.d_right)
 
        self.W_in_left = 2 * (self.random_uniform((self.d_left, self.N_in_left)) - 0.5) # TODO: check if I should make this complex later
        self.W_in_left = tf.cast(self.W_in_left, self.dtype)
        self.W_in_right = 2 * (self.random_uniform((self.N_in_right, self.d_right)) - 0.5) # TODO: check if I should make this complex later
        self.W_in_right = tf.cast(self.W_in_right, self.dtype)

        # TODO: using a vectorization trick to learn one vectorized W_out instead of left and right W_outs.
        # This is mathematically equivalent to 
        # self.W_out_left = self.RS.randn(self.N_r, self.d_left)
        # self.W_out_right = self.RS.randn(self.d_right + self.N_in_right, self.N_t)
        self.feature_dim = int(self.d_left * self.d_right * (self.window_length + 1))

        w_out_shape = (self.N_r * self.N_t, self.feature_dim)
        w_real = self.rng.normal(shape=w_out_shape, dtype=self.real_dtype)
        w_imag = self.rng.normal(shape=w_out_shape, dtype=self.real_dtype)

        scale = tf.cast(1.0 / tf.sqrt(2.0), self.real_dtype)
        self.W_out = tf.complex(scale * w_real, scale * w_imag)

        self.S_0 = tf.zeros([self.d_left, self.d_right], dtype=self.dtype)
    
    def predict(self, h_freq_csi_history):

        h_freq_csi_predicted = self.pred_v2(h_freq_csi_history)

        return h_freq_csi_predicted
    

    def pred_v2(self, h_freq_csi_history):
        
        
        tf.debugging.assert_equal(tf.rank(h_freq_csi_history), 8, message="\n The dimensions of h_freq_csi_history are not correct")
        h_freq_csi_history = tf.transpose(h_freq_csi_history, perm=[0,1,2,3,4,5,7,6])
        shape_tensor = tf.shape(h_freq_csi_history)
        num_batches = int(shape_tensor[1])
        num_rx_nodes = int(shape_tensor[2])
        num_rx_antennas = int(shape_tensor[3])
        num_tx_nodes = int(shape_tensor[4])
        num_tx_antennas = int(shape_tensor[5])
        num_freq_res = int(shape_tensor[6])
        num_ofdm_syms = int(shape_tensor[7])

        channel_train_input = h_freq_csi_history[:-1, ...]
        channel_train_gt    = h_freq_csi_history[1:,  ...]
        
        if not self.enable_window:
            window_weights = None

        chan_pred_var = tf.Variable(tf.zeros_like(h_freq_csi_history[0, ...], dtype=self.dtype))

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
                        # self.S_0 = tf.zeros([self.d_left, self.d_right], dtype=self.dtype)

                        S_f, Y_f = self.build_S_Y(Y_in, Y_out, curr_window_weights=None)
                        S_list.append(S_f); Y_list.append(Y_f)

                S_all = tf.concat(S_list, axis=1)  # (F, sum_T)
                Y_all = tf.concat(Y_list, axis=1)  # (N_r*N_t, sum_T)

                # --------- (B) SINGLE READOUT SOLVE (shared across RBs) ----------
                # Prefer ridge for stability:
                G = self.reg_p_inv(S_all)               # (sum_T, F)  :=  S_all^H (S_all S_all^H + λI)^{-1}
                self.W_out = tf.matmul(Y_all, G)        # (N_r*N_t, F)

                # --------- (C) PREDICTION PHASE with the shared W_out ----------
                for freq_re in range(num_freq_res):
                    for ofdm_sym in range(num_ofdm_syms):
                        # Use last known channel as test input; predict next step
                        channel_test_input = channel_train_gt[:, 0, tx_node, :, rx_node, :, freq_re, ofdm_sym]

                        # Optional: either carry S_0 across RBs for smoothness,
                        # or reset it per RB. Start with reset; then try carry-over.
                        self.S_0 = tf.zeros([self.d_left, self.d_right], dtype=self.dtype)

                        channel_pred_temp = self.test_train_predict(channel_test_input, curr_window_weights=None)
                        channel_pred_temp = channel_pred_temp[:, :, -1:]       # keep last step
                        channel_pred_temp = tf.squeeze(channel_pred_temp)      # [N_r, N_t]
                        chan_pred_var[:, tx_node, :, rx_node, :, freq_re, ofdm_sym].assign(channel_pred_temp)

        chan_pred = tf.transpose(chan_pred_var.read_value(), perm=[0,1,2,3,4,6,5])
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
            padding = tf.zeros([tf.shape(Y_3D)[0], forget, tf.shape(Y_3D)[2]], dtype=self.dtype)
            Y_3D_win = tf.concat([Y_3D, padding], axis=1)

        S_3D_transit = self.state_transit(Y_3D_win * self.input_scale)
        S_3D = tf.concat([S_3D_transit, Y_3D_win], axis=-1)

        T = tf.shape(S_3D)[0]
        S = tf.stack([tf.reshape(S_3D[t], [-1]) for t in tf.range(T)], axis=1)  # (feature_dim, T)
        Y = tf.stack([tf.reshape(Y_target_3D[t], [-1]) for t in tf.range(T)], axis=1)  # (N_r*N_t, T)
        
        return S, Y

    def sparse_mat(self, m):
        
        real_part = self.rng.uniform(shape=(m, m), minval=-1.0, maxval=1.0, dtype=self.real_dtype)
        imag_part = self.rng.uniform(shape=(m, m), minval=-1.0, maxval=1.0, dtype=self.real_dtype)
        W = tf.complex(real_part, imag_part)
        mask = self.rng.uniform(shape=(m, m), dtype=self.real_dtype) < self.sparsity
        W = tf.where(mask, tf.zeros_like(W), W)
        radius = tf.cast(tf.reduce_max(tf.abs(tf.linalg.eigvals(W))), self.dtype)
        W = tf.where(tf.equal(radius, 0), W, W * (self.spectral_radius / radius))
        
        return W

    def complex_to_real_target(self, Y_target_2D):
        Y_target_2D_real_list = []
        for t in range(self.N_t):
            target = tf.reshape(Y_target_2D[t, :], [1, -1]) # (1, N_symbols * (N_fft+N_cp))
            real_target = tf.concat((tf.math.real(target), tf.math.imag(target)), axis=0)  # (2, N_symbols * (N_fft+N_cp))
            Y_target_2D_real_list.append(real_target)
        Y_target_2D_real = tf.concat(Y_target_2D_real_list, axis=0)
        
        return Y_target_2D_real

    def fitting_time(self, channel_input, channel_output, curr_window_weights):

        Y_3D = channel_input
        Y_target_3D = channel_output

        if self.enable_window:
            Y_3D_new = self.form_window_input_signal(Y_3D, curr_window_weights)
        else:
            # TODO: not sure if this still works. add in forget length functionality later
            padding = tf.zeros([tf.shape(Y_3D)[0], self.forget_length], dtype=self.dtype)
            Y_3D_new = tf.concat([Y_3D, padding], axis=1)

        S_3D_transit = self.state_transit(Y_3D_new * self.input_scale)

        S_3D = tf.concat([S_3D_transit, Y_3D_new], axis=-1)

        T = tf.shape(S_3D)[0]
        S = tf.stack([
            tf.reshape(S_3D[t], [-1]) for t in tf.range(T)
        ], axis=1)  # (feature_dim, T)

        # vectorization trick. equivalent to having two W_out matrices on either side of the feature matrix being fed to the output
        Y = tf.stack([
            tf.reshape(Y_target_3D[t], [-1]) for t in tf.range(T)
        ], axis=1)

        self.W_out = tf.matmul(Y, tf.linalg.pinv(S))

        pred_channel = tf.matmul(self.W_out, S)

        pred_channel = tf.reshape(pred_channel, [self.N_r, self.N_t, -1])

        return pred_channel

    def cal_nmse(self, H, H_hat):
        H_hat = tf.cast(H_hat, dtype=H.dtype)
        mse = tf.reduce_sum(tf.abs(H - H_hat) ** 2)
        normalization_factor = tf.reduce_sum((tf.abs(H) + tf.abs(H_hat)) ** 2)

        nmse = mse / normalization_factor
        return nmse

    def reg_p_inv(self, X):
        # X: (F, T)
        F = tf.shape(X)[0]
        eye = tf.eye(F, dtype=self.dtype)
        G = tf.matmul(X, X, adjoint_b=True) + self.reg * eye  # (F,F)
        G = tf.matmul(tf.transpose(X, conjugate=True), self.complex_pinv(G))                 # (T,F)

        return G
    
    def complex_pinv(self, A, rcond=1e-6):
        """
        Moore–Penrose pseudoinverse that works for real or complex A.

        A: [..., M, N]
        returns: [..., N, M]
        """
        # SVD: A = U diag(s) Vᴴ
        s, u, v = tf.linalg.svd(A, full_matrices=False)  # s: [..., k], u: [..., M, k], v: [..., N, k]
        # Invert singular values with cutoff
        s_max = tf.reduce_max(s, axis=-1, keepdims=True)
        cutoff = rcond * s_max
        s_inv = tf.where(s > cutoff, 1.0 / s, tf.zeros_like(s))  # [..., k]

        # Build diag(1/s) and compute A⁺ = V diag(1/s) Uᴴ
        s_inv_mat = tf.linalg.diag(s_inv)                        # [..., k, k]
        s_inv_mat = tf.cast(s_inv_mat, self.dtype)
        A_pinv = v @ s_inv_mat @ tf.linalg.adjoint(u)            # [..., N, M]
        return A_pinv

    def form_window_input_signal(self, Y_3D_complex, curr_window_weights):

        tf.debugging.assert_equal(tf.rank(Y_3D_complex), 3, message="Y must be [T, N_r, N_t]")
        shape_tensor = tf.shape(Y_3D_complex)
        T = int(shape_tensor[0])
        N_r = int(shape_tensor[1])
        N_t = int(shape_tensor[2])

        L = int(self.window_length)

        Y_3D_window = tf.TensorArray(self.dtype, size=T)

        zero_block = tf.zeros((N_r, N_t), dtype=self.dtype)

        for k in range(T):
            blocks = []
            for ell in range(L):
                t = k - ell
                if t >= 0:
                    blocks.append(Y_3D_complex[t])       # [N_r, N_t]
                else:
                    blocks.append(zero_block)  # causal zero-pad
            Y_3D_window = Y_3D_window.write(k, tf.concat(blocks, axis=-1))

        return Y_3D_window.stack()

    def test_train_predict(self, channel_train_input, curr_window_weights):
        self.S_0 = tf.zeros([self.d_left, self.d_right], dtype=self.dtype)

        Y_3D_org = channel_train_input

        Y_3D = self.form_window_input_signal(Y_3D_org, curr_window_weights)

        S_3D = self.state_transit(Y_3D * self.input_scale)

        S_3D = tf.concat([S_3D, Y_3D], axis=-1)

        # vectorization trick. equivalent to having two W_out matrices on either side of the feature matrix being fed to the output
        T = tf.shape(S_3D)[0]
        S = tf.stack([
            tf.reshape(S_3D[t], [-1]) for t in tf.range(T)
        ], axis=1)  # (feature_dim, T)

        curr_channel_pred = tf.matmul(self.W_out, S)

        curr_channel_pred = tf.reshape(curr_channel_pred, [self.N_r, self.N_t, -1])

        return curr_channel_pred

    def state_transit(self, Y_3D):

        T = tf.shape(Y_3D)[0] # number of samples
        
        S_2D = tf.identity(self.S_0)
        S_list = []
        for t in tf.range(T):
            S_2D = self.complex_tanh(tf.matmul(tf.matmul(self.W_res_left, S_2D), self.W_res_right) + tf.matmul(tf.matmul(self.W_in_left, Y_3D[t,:,:]), self.W_in_right))
            S_list.append(S_2D)

        S_3D = tf.stack(S_list, axis=0)

        self.S_0 = S_2D

        return S_3D

    def complex_tanh(self, Y):
        return tf.complex(tf.math.tanh(tf.math.real(Y)), tf.math.tanh(tf.math.imag(Y)))