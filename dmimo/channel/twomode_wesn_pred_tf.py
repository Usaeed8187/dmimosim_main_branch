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
        self.feature_dim = int(self.d_left * (self.d_right + self.N_in_right))

        w_out_shape = (self.N_r * self.N_t, self.feature_dim)
        w_real = self.rng.normal(shape=w_out_shape, dtype=self.real_dtype)
        w_imag = self.rng.normal(shape=w_out_shape, dtype=self.real_dtype)

        scale = tf.cast(1.0 / tf.sqrt(2.0), self.real_dtype)
        self.W_out = tf.complex(scale * w_real, scale * w_imag)

        self.S_0 = tf.zeros([self.d_left, self.d_right], dtype=self.dtype)
    
    @tf.function
    def predict(self, h_freq_csi_history):

        h_freq_csi_predicted = self.pred_v2(h_freq_csi_history)

        return h_freq_csi_predicted

    @tf.function
    def pred_v2(self, h_freq_csi_history):
        
        tf.debugging.assert_equal(tf.rank(h_freq_csi_history), 8, message="\n The dimensions of h_freq_csi_history are not correct")
        h_freq_csi_history = tf.transpose(h_freq_csi_history, perm=[0,1,2,3,4,5,7,6])
        shape_tensor = tf.shape(h_freq_csi_history)
        num_batches = shape_tensor[1]
        num_tx_nodes = shape_tensor[2]
        num_rx_antennas = shape_tensor[3]
        num_rx_nodes = shape_tensor[4]
        num_tx_antennas = shape_tensor[5]
        num_freq_res = shape_tensor[6]
        num_ofdm_syms = shape_tensor[7]

        channel_train_input = h_freq_csi_history[:-1, ...]
        channel_train_gt = h_freq_csi_history[1:, ...]

        channel_train_input = tf.reshape(
            channel_train_input,
            [
                tf.shape(channel_train_input)[0],
                num_batches,
                num_tx_nodes,
                num_rx_antennas,
                num_rx_nodes,
                num_tx_antennas,
                num_freq_res * num_ofdm_syms,
            ],
        )
        channel_train_gt = tf.reshape(
            channel_train_gt,
            [
                tf.shape(channel_train_gt)[0],
                num_batches,
                num_tx_nodes,
                num_rx_antennas,
                num_rx_nodes,
                num_tx_antennas,
                num_freq_res * num_ofdm_syms,
            ],
        )

        combined_count = tf.shape(channel_train_input)[-1]
        total_pairs = num_rx_nodes * num_tx_nodes

        def process_pair(pair_idx):
            rx_node = pair_idx // num_tx_nodes
            tx_node = pair_idx % num_tx_nodes

            self.init_weights()

            train_input_pair = channel_train_input[:, 0, tx_node, :, rx_node, :, :]
            train_gt_pair = channel_train_gt[:, 0, tx_node, :, rx_node, :, :]

            def build_features(idx):
                S_f, Y_f = self.build_S_Y(
                    train_input_pair[..., idx], train_gt_pair[..., idx], curr_window_weights=None
                )
                return S_f, Y_f

            S_stack, Y_stack = tf.map_fn(
                build_features,
                tf.range(combined_count),
                fn_output_signature=(
                    tf.TensorSpec(shape=(self.feature_dim, None), dtype=self.dtype),
                    tf.TensorSpec(shape=(self.N_r * self.N_t, None), dtype=self.dtype),
                ),
            )

            S_all = tf.reshape(tf.transpose(S_stack, perm=[1, 0, 2]), [self.feature_dim, -1])
            Y_all = tf.reshape(
                tf.transpose(Y_stack, perm=[1, 0, 2]), [self.N_r * self.N_t, -1]
            )

            G = self.reg_p_inv(S_all)
            self.W_out = tf.matmul(Y_all, G)

            def predict_block(idx):
                channel_test_input = train_gt_pair[..., idx]
                channel_pred_temp = self.test_train_predict(channel_test_input, curr_window_weights=None)
                return channel_pred_temp[:, :, -1]

            pair_pred = tf.map_fn(
                predict_block,
                tf.range(combined_count),
                fn_output_signature=tf.TensorSpec(
                    shape=(self.N_r, self.N_t), dtype=self.dtype
                ),
            )

            return pair_pred

        pair_predictions = tf.map_fn(
            process_pair,
            tf.range(total_pairs),
            fn_output_signature=tf.TensorSpec(
                shape=(None, self.N_r, self.N_t), dtype=self.dtype
            ),
        )

        pair_predictions = tf.reshape(
            pair_predictions,
            [
                num_rx_nodes,
                num_tx_nodes,
                num_freq_res,
                num_ofdm_syms,
                self.N_r,
                self.N_t,
            ],
        )

        pair_predictions = tf.transpose(pair_predictions, perm=[1, 4, 0, 5, 2, 3])
        pair_predictions = tf.expand_dims(pair_predictions, axis=0)

        chan_pred = tf.transpose(pair_predictions, perm=[0, 1, 2, 3, 4, 6, 5])
        
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
        S = tf.transpose(tf.reshape(S_3D, [T, -1]))  # (feature_dim, T)
        Y = tf.transpose(tf.reshape(Y_target_3D, [T, -1]))  # (N_r*N_t, T)
        
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
        S = tf.transpose(tf.reshape(S_3D, [T, -1]))  # (feature_dim, T)

        # vectorization trick. equivalent to having two W_out matrices on either side of the feature matrix being fed to the output
        Y = tf.transpose(tf.reshape(Y_target_3D, [T, -1]))

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
        T = shape_tensor[0]
        N_r = shape_tensor[1]
        N_t = shape_tensor[2]

        L = tf.cast(self.window_length, tf.int32)

        zero_pad = tf.zeros((tf.maximum(L - 1, 0), N_r, N_t), dtype=self.dtype)
        padded = tf.concat([zero_pad, Y_3D_complex], axis=0)

        time_indices = tf.range(T)[:, None] + tf.range(L)[None, :]
        windows = tf.gather(padded, time_indices, axis=0)  # [T, L, N_r, N_t]

        windows = tf.transpose(windows, perm=[0, 2, 3, 1])
        Y_3D_window = tf.reshape(windows, [T, N_r, N_t * L])

        return Y_3D_window

    def test_train_predict(self, channel_train_input, curr_window_weights):
        self.S_0 = tf.zeros([self.d_left, self.d_right], dtype=self.dtype)

        Y_3D_org = channel_train_input

        Y_3D = self.form_window_input_signal(Y_3D_org, curr_window_weights)

        S_3D = self.state_transit(Y_3D * self.input_scale)

        S_3D = tf.concat([S_3D, Y_3D], axis=-1)

        # vectorization trick. equivalent to having two W_out matrices on either side of the feature matrix being fed to the output
        T = tf.shape(S_3D)[0]
        S = tf.transpose(tf.reshape(S_3D, [T, -1]))  # (feature_dim, T)

        curr_channel_pred = tf.matmul(self.W_out, S)

        curr_channel_pred = tf.reshape(curr_channel_pred, [self.N_r, self.N_t, -1])

        return curr_channel_pred

    def state_transit(self, Y_3D):

        T = tf.shape(Y_3D)[0] # number of samples
        
        def step(prev_state, curr_input):
            update = tf.matmul(tf.matmul(self.W_res_left, prev_state), self.W_res_right)
            update += tf.matmul(tf.matmul(self.W_in_left, curr_input), self.W_in_right)
            return self.complex_tanh(update)

        S_3D = tf.scan(step, Y_3D, initializer=self.S_0)
        self.S_0 = S_3D[-1]

        return S_3D

    def complex_tanh(self, Y):
        return tf.complex(tf.math.tanh(tf.math.real(Y)), tf.math.tanh(tf.math.imag(Y)))
    
def predict_all_links_tf(h_freq_csi_history, rc_config, ns3cfg, num_bs_ant=4, num_ue_ant=2):

    _, _, _, _, _, _, _, RB = h_freq_csi_history.shape

    # Convert once to TensorFlow tensor to avoid repeated host/device transfers
    tf_h_freq_csi_history = tf.convert_to_tensor(h_freq_csi_history)
    h_freq_csi = tf.Variable(
        tf.zeros(tf_h_freq_csi_history[0, ...].shape, dtype=tf_h_freq_csi_history.dtype)
    )

    # Pre-compute antenna index ranges for all TX/RX nodes
    tx_ant_ranges = [tf.range(0, num_bs_ant)] + [
        tf.range(num_bs_ant + (idx - 1) * num_ue_ant, num_bs_ant + idx * num_ue_ant)
        for idx in range(1, ns3cfg.num_txue_sel + 1)
    ]
    rx_ant_ranges = [tf.range(0, num_bs_ant)] + [
        tf.range(num_bs_ant + (idx - 1) * num_ue_ant, num_bs_ant + idx * num_ue_ant)
        for idx in range(1, ns3cfg.num_rxue_sel + 1)
    ]

    predictor_cache = {}

    for tx_ant_idx in tx_ant_ranges:
        for rx_ant_idx in rx_ant_ranges:
            rx_start, rx_len = int(rx_ant_idx[0]), int(rx_ant_idx.shape[0])
            tx_start, tx_len = int(tx_ant_idx[0]), int(tx_ant_idx.shape[0])

            curr_h_freq_csi_history = tf.gather(tf_h_freq_csi_history, rx_ant_idx, axis=3)
            curr_h_freq_csi_history = tf.gather(curr_h_freq_csi_history, tx_ant_idx, axis=5)

            key = (rx_len, tx_len)
            if key not in predictor_cache:
                predictor_cache[key] = twomode_wesn_pred_tf(
                    rc_config=rc_config,
                    num_freq_re=RB,
                    num_rx_ant=rx_len,
                    num_tx_ant=tx_len,
                )

            tmp = tf.convert_to_tensor(predictor_cache[key].predict(curr_h_freq_csi_history))

            h_freq_csi[:, :, rx_start : rx_start + rx_len, :, tx_start : tx_start + tx_len, :, :].assign(tmp)


    return h_freq_csi