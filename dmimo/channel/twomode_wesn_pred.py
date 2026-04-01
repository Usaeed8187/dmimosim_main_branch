import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation
from dmimo.channel.kalman_filter_pred import kalman_filter_pred

class twomode_wesn_pred:

    def __init__(self,
                rc_config, 
                num_freq_re,
                num_rx_ant,
                num_tx_ant, 
                readout_solve_method="vectorization_trick",
                windowing_mode='col_concat',
                state_dim_setting='from_config',
                type=np.complex64):
        
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
        self.enable_kalman_weight_config = bool(getattr(rc_config, "enable_kalman_weight_config", False))
        self.kalman_weight_ar_order = rc_config.window_length
        self.kalman_gain_iters = int(getattr(rc_config, "kalman_gain_iters", 100))
        self.kalman_eps = float(getattr(rc_config, "kalman_eps", 1e-8))
        self.readout_solve_method = readout_solve_method # "vectorization_trick", "ALS"
        self.kalman_project_to_bilinear = bool(
            getattr(rc_config, "kalman_project_to_bilinear", self.readout_solve_method == "ALS")
        )

        self.windowing_mode = windowing_mode # "col_concat", "block_diag"
        if self.windowing_mode not in ("col_concat", "block_diag"):
            raise ValueError(
                f"Unsupported windowing_mode='{self.windowing_mode}'. "
                "Use 'col_concat' or 'block_diag'."
            )
        self.state_dim_setting = state_dim_setting
        if self.state_dim_setting not in ("from_data", "from_config"):
            raise ValueError(
                f"Unsupported state_dim_setting='{self.state_dim_setting}'. "
                "Use 'from_data' or 'from_config'."
            )

        seed = 10
        self.RS = np.random.RandomState(seed)

        if self.enable_window:
            if self.windowing_mode == "block_diag":
                self.N_in_left = self.N_r * self.window_length
                self.N_in_right = self.N_t * self.window_length
            else:
                self.N_in_left = self.N_r
                self.N_in_right = self.N_t * self.window_length
        else:
            self.N_in_left = self.N_r
            self.N_in_right = self.N_t

        if self.state_dim_setting == "from_data":
            self.d_left = self.N_in_left
            self.d_right = self.N_in_right
        else:
            self.d_left = getattr(rc_config, "state_dim_left", None)
            self.d_right = getattr(rc_config, "state_dim_right", None)
            if self.d_left is None or self.d_right is None:
                raise ValueError(
                    "state_dim_setting='from_config' requires both "
                    "rc_config.state_dim_left and rc_config.state_dim_right."
                )
            self.d_left = int(self.d_left)
            self.d_right = int(self.d_right)
            if self.d_left <= 0 or self.d_right <= 0:
                raise ValueError(
                    "state_dim_left and state_dim_right must be positive integers."
                )

        self.init_weights()

    def init_weights(self):

        self.W_res_left = self.sparse_mat(self.d_left)
        self.W_res_right = self.sparse_mat(self.d_right)

        self.W_in_left = 2 * (self.RS.rand(self.d_left, self.N_in_left) - 0.5) # TODO: check if I should make this complex later
        self.W_in_right = 2 * (self.RS.rand(self.N_in_right, self.d_right) - 0.5) # TODO: check if I should make this complex later

        self.state_feature_dim = int(self.d_left * self.d_right)
        self.input_feature_dim = int(self.N_in_left * self.N_in_right)
        self.feature_dim = int(self.state_feature_dim + self.input_feature_dim)
        self.feature_left_dim = int(self.d_left + self.N_in_left)
        self.feature_right_dim = int(self.d_right + self.N_in_right)

        # vectorization readout (legacy/default path)
        self.W_out = self.RS.randn(self.N_r * self.N_t, self.feature_dim).astype(self.dtype)
        # bilinear readout (ALS path)
        self.W_out_left = self.RS.randn(self.N_r, self.feature_left_dim).astype(self.dtype)
        self.W_out_right = self.RS.randn(self.feature_right_dim, self.N_t).astype(self.dtype)

        self.S_0 = np.zeros([self.d_left, self.d_right], dtype=self.dtype)
        self.W_res_kron = None
        self.W_in_kron = None
        self.use_kalman_bilinear = False

    
    def predict(self, h_freq_csi_history, err_var_history=None):

        if self.enable_kalman_weight_config and err_var_history is None:
            warnings.warn(
                "enable_kalman_weight_config=True but err_var_history was not provided. "
                "Falling back to heuristic observation-noise covariance R.",
                RuntimeWarning,
            )

        h_freq_csi_predicted = self.pred(h_freq_csi_history, err_var_history=err_var_history)

        return h_freq_csi_predicted
    
    def pred(self, h_freq_csi_history, err_var_history=None):
        
        h_freq_csi_history = np.asarray(h_freq_csi_history)
        err_var_hist_aligned = None
        if err_var_history is not None:
            err_var_hist_aligned = np.asarray(err_var_history)
        if h_freq_csi_history.ndim == 8:
            h_freq_csi_history = h_freq_csi_history.transpose([0,1,2,3,4,5,7,6])
            if err_var_hist_aligned is not None:
                err_var_hist_aligned = err_var_hist_aligned.transpose([0,1,2,3,4,5,7,6])
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
        err_var_train_input = None
        if err_var_hist_aligned is not None:
            err_var_train_input = err_var_hist_aligned[:-1, ...]
        
        if not self.enable_window:
            window_weights = None

        chan_pred = np.zeros(h_freq_csi_history[0,...].shape, dtype=self.dtype)

        # === ONE reservoir per (rx_node, tx_node) pair; shared across all RBs ===
        for rx_node in range(num_rx_nodes):
            for tx_node in range(num_tx_nodes):

                # Initialize weights ONCE for all RBs of this (rx_node, tx_node)
                self.init_weights()

                # --------- (A) FEATURE BUILD PHASE: stack all RBs (and OFDM syms) ----------
                # Flatten (freq_re, ofdm_sym) into a single axis and build one long sequence:
                # [T, N_r, N_t, F, O] -> [F, O, T, N_r, N_t] -> [F*O*T, N_r, N_t]
                Y_in_all = channel_train_input[:, 0, tx_node, :, rx_node, :, :, :]
                Y_out_all = channel_train_gt[:, 0, tx_node, :, rx_node, :, :, :]

                T_train = Y_in_all.shape[0]
                B_fo = num_freq_res * num_ofdm_syms

                Y_in_batch = np.transpose(Y_in_all, (0, 3, 4, 1, 2)).reshape(
                    T_train, B_fo, self.N_r, self.N_t
                )
                Y_out_batch = np.transpose(Y_out_all, (0, 3, 4, 1, 2)).reshape(
                    T_train, B_fo, self.N_r, self.N_t
                )

                if self.enable_kalman_weight_config:
                    E_in_seq = None
                    if err_var_train_input is not None:
                        E_in_all = err_var_train_input[:, 0, tx_node, :, rx_node, :, :, :]
                        E_in_seq = np.transpose(E_in_all, (0, 3, 4, 1, 2)).reshape(
                            T_train, B_fo, self.N_r, self.N_t
                        )
                    self.configure_weights_from_kalman(Y_in_batch, curr_window_weights=None, err_var_input=E_in_seq)

                S_all, Y_all = self.build_S_Y(Y_in_batch, Y_out_batch, curr_window_weights=None)

                # --------- (B) SINGLE READOUT SOLVE (shared across RBs) ----------
                if self.readout_solve_method == "ALS":
                    X_all = self._feature_matrices_from_flat(S_all)
                    Y_all_mats = self._target_matrices_from_flat(Y_all)
                    self.W_out_left, self.W_out_right = self.solve_readout_als(X_all, Y_all_mats)
                elif self.readout_solve_method == "vectorization_trick":
                    # Prefer ridge for stability:
                    G = self.reg_p_inv(S_all)               # (sum_T, F)  :=  S_all^H (S_all S_all^H + λI)^{-1}
                    self.W_out = Y_all @ G                  # (N_r*N_t, F)
                else:
                    raise ValueError(
                        f"Unsupported readout_solve_method='{self.readout_solve_method}'. "
                        "Use 'vectorization_trick' or 'ALS'."
                    )

                # --------- (C) PREDICTION PHASE with the shared W_out ----------
                # Predict all (freq_re, ofdm_sym) streams in one batched pass.
                channel_test_input_all = channel_train_gt[:, 0, tx_node, :, rx_node, :, :, :]
                T_test = channel_test_input_all.shape[0]
                B_fo = num_freq_res * num_ofdm_syms

                # [T, N_r, N_t, F, O] -> [T, F, O, N_r, N_t] -> [T, B_fo, N_r, N_t]
                channel_test_input_batch = np.transpose(channel_test_input_all, (0, 3, 4, 1, 2)).reshape(
                    T_test, B_fo, self.N_r, self.N_t
                )

                channel_pred_batch = self.test_train_predict_batch(channel_test_input_batch, curr_window_weights=None)
                channel_pred_last = channel_pred_batch[:, :, :, -1]  # [B_fo, N_r, N_t]

                # [B_fo, N_r, N_t] -> [F, O, N_r, N_t] -> [N_r, N_t, F, O]
                channel_pred_fo = channel_pred_last.reshape(num_freq_res, num_ofdm_syms, self.N_r, self.N_t)
                channel_pred_fo = np.transpose(channel_pred_fo, (2, 3, 0, 1))
                chan_pred[:, tx_node, :, rx_node, :, :, :] = channel_pred_fo

        chan_pred = chan_pred.transpose([0,1,2,3,4,6,5])
        return chan_pred

    def build_S_Y(self, channel_input, channel_output, curr_window_weights):
        Y_4D, _ = self._ensure_batch_axis(channel_input)
        Y_target_4D, _ = self._ensure_batch_axis(channel_output)
        Y_4D = self._prepare_inputs(Y_4D, curr_window_weights)

        S_4D = self._state_transit_core(Y_4D * self.input_scale, reset_state=True)
        S = self._flatten_features(S_4D, Y_4D)
        Y = self._flatten_targets(Y_target_4D)
        return S, Y

    def sparse_mat(self, m):
        
        max_retries = 5
        for _ in range(max_retries):
            W = 2 * (self.RS.rand(m, m) - 0.5) + 2j * (self.RS.rand(m, m) - 0.5)
            W[self.RS.rand(*W.shape) < self.sparsity] = 0.0 + 0.0j

            radius = float(np.max(np.abs(np.linalg.eigvals(W))))
            if np.isfinite(radius) and radius > 1e-12:
                W = W * (self.spectral_radius / radius)
                return W.astype(self.dtype)

        raise ValueError("\nSpectral radius = 0")

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
        S = self._flatten_features(S_3D_transit[:, None, :, :], Y_3D_new[:, None, :, :])
        T = Y_target_3D.shape[0]
        
        Y = np.column_stack([
            Y_target_3D[t].reshape(-1, order='C') for t in range(T)
        ])
        
        if self.readout_solve_method == "ALS":
            X_samples = self._feature_matrices_from_flat(S)
            Y_samples = self._target_matrices_from_flat(Y)
            self.W_out_left, self.W_out_right = self.solve_readout_als(X_samples, Y_samples)
            pred_stack = np.array(
                [self.W_out_left @ X_samples[i] @ self.W_out_right for i in range(X_samples.shape[0])],
                dtype=self.dtype,
            )
            pred_channel = np.transpose(pred_stack, (1, 2, 0))
        else:
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

    def _ensure_batch_axis(self, Y):
        Y = np.asarray(Y)
        if Y.ndim == 3:
            return Y[:, None, :, :], False
        if Y.ndim == 4:
            return Y, True
        raise ValueError("Input must be [T, N_r, N_t] or [T, B, N_r, N_t]")
    
    def _form_window_input_signal_core(self, Y_4D_complex, curr_window_weights):
        # Y: [T, B, N_r, N_t]
        T, B, N_r, N_t = Y_4D_complex.shape
        L = int(self.window_length)

        if self.windowing_mode == "col_concat":
            Y_4D_window = np.zeros((T, B, N_r, L * N_t), dtype=self.dtype)
        elif self.windowing_mode == "block_diag":
            Y_4D_window = np.zeros((T, B, L * N_r, L * N_t), dtype=self.dtype)
        else:
            raise ValueError(
                f"Unsupported windowing_mode='{self.windowing_mode}'. "
                "Use 'col_concat' or 'block_diag'."
            )

        for k in range(T):
            blocks = []
            for ell in range(L):
                t = k - ell
                if t >= 0:
                    blocks.append(Y_4D_complex[t])       # [B, N_r, N_t]
                else:
                    blocks.append(np.zeros((B, N_r, N_t), dtype=self.dtype))
            if self.windowing_mode == "col_concat":
                Y_4D_window[k] = np.concatenate(blocks, axis=-1)
            else:
                for ell, blk in enumerate(blocks):
                    r0 = ell * N_r
                    c0 = ell * N_t
                    Y_4D_window[k, :, r0:r0 + N_r, c0:c0 + N_t] = blk

        return Y_4D_window

    def _prepare_inputs(self, Y_4D_org, curr_window_weights):
        if self.enable_window:
            return self._form_window_input_signal_core(Y_4D_org, curr_window_weights)

        forget = getattr(self, "forget_length", 0)
        if forget == 0:
            return Y_4D_org

        return np.concatenate(
            [
                Y_4D_org,
                np.zeros([Y_4D_org.shape[0], Y_4D_org.shape[1], forget, Y_4D_org.shape[3]], dtype=self.dtype),
            ],
            axis=2,
        )

    def _state_transit_core(self, Y_4D, reset_state):
        T, B = Y_4D.shape[0], Y_4D.shape[1]
        if reset_state:
            S_3D = np.zeros([B, self.d_left, self.d_right], dtype=self.dtype)
        else:
            S_3D = np.broadcast_to(self.S_0, (B, self.d_left, self.d_right)).copy()

        S_4D = []
        for t in range(T):
            if (
                self.enable_kalman_weight_config
                and (not self.use_kalman_bilinear)
                and self.W_res_kron is not None
                and self.W_in_kron is not None
            ):
                S_vec = S_3D.reshape(B, -1)
                U_vec = Y_4D[t].reshape(B, -1)
                next_vec = (S_vec @ self.W_res_kron.T) + (U_vec @ self.W_in_kron.T)
                S_3D = self.complex_tanh(next_vec.reshape(B, self.d_left, self.d_right))
            else:
                recurrent_term = (self.W_res_left @ S_3D) @ self.W_res_right
                input_term = (self.W_in_left @ Y_4D[t]) @ self.W_in_right
                S_3D = self.complex_tanh(recurrent_term + input_term)

            S_4D.append(S_3D)

        S_4D = np.stack(S_4D, axis=0)

        self.S_0 = S_3D[0]

        return S_4D

    def _flatten_features(self, S_4D, Y_4D):
        T, B = S_4D.shape[:2]
        S_flat = S_4D.reshape(T * B, self.state_feature_dim)
        Y_flat = Y_4D.reshape(T * B, self.input_feature_dim)
        return np.concatenate([S_flat, Y_flat], axis=1).T
    
    def _flatten_targets(self, Y_4D):
        T, B = Y_4D.shape[:2]
        return Y_4D.reshape(T * B, -1).T

    def _feature_matrices_from_flat(self, S_flat):
        # S_flat: [feature_dim, N_samples] -> X: [N_samples, d_left+N_in_left, d_right+N_in_right]
        S_samples = np.asarray(S_flat).T
        state_samples = S_samples[:, :self.state_feature_dim].reshape(-1, self.d_left, self.d_right)
        input_samples = S_samples[:, self.state_feature_dim:].reshape(-1, self.N_in_left, self.N_in_right)
        return self._compose_feature_matrices(state_samples, input_samples)

    def _compose_feature_matrices(self, state_samples, input_samples):
        n_samples = state_samples.shape[0]
        X = np.zeros(
            (n_samples, self.feature_left_dim, self.feature_right_dim),
            dtype=self.dtype,
        )
        X[:, :self.d_left, :self.d_right] = state_samples
        X[:, self.d_left:, self.d_right:] = input_samples
        return X

    def _feature_matrices_from_state_input(self, S_4D, Y_4D):
        T, B = S_4D.shape[:2]
        state_samples = S_4D.reshape(T * B, self.d_left, self.d_right)
        input_samples = Y_4D.reshape(T * B, self.N_in_left, self.N_in_right)
        return self._compose_feature_matrices(state_samples, input_samples)


    def _target_matrices_from_flat(self, Y_flat):
        # Y_flat: [N_r*N_t, N_samples] -> Y: [N_samples, N_r, N_t]
        Y_samples = np.asarray(Y_flat).T
        return Y_samples.reshape(-1, self.N_r, self.N_t)

    def _ridge_left_solve(self, Y_cat, X_cat):
        # Solve min_W ||Y - W X||_F^2 + reg||W||_F^2
        XXH = X_cat @ X_cat.conj().T + self.reg * np.eye(X_cat.shape[0], dtype=self.dtype)
        return (Y_cat @ X_cat.conj().T) @ np.linalg.pinv(XXH)

    def solve_readout_als(self, X_samples, Y_samples, max_iters=25, tol=1e-5):
        # X_samples: [N, feature_left_dim, feature_right_dim], Y_samples: [N, N_r, N_t]
        W_left = self.W_out_left.copy()
        W_right = self.W_out_right.copy()

        prev_rel_err = np.inf
        for _ in range(max_iters):
            # Update left: Y_i ~= W_left (X_i W_right)
            Z_list = [X_samples[i] @ W_right for i in range(X_samples.shape[0])]
            Z_cat = np.concatenate(Z_list, axis=1)           # [d_left, N*N_t]
            Y_cat = np.concatenate([Y_samples[i] for i in range(Y_samples.shape[0])], axis=1)  # [N_r, N*N_t]
            W_left = self._ridge_left_solve(Y_cat, Z_cat).astype(self.dtype)

            # Update right via transpose: Y_i^T ~= W_right^T (W_left X_i)^T
            A_list = [W_left @ X_samples[i] for i in range(X_samples.shape[0])]
            A_cat_t = np.concatenate([A.T for A in A_list], axis=1)                             # [d_right+N_in_right, N*N_r]
            Y_cat_t = np.concatenate([Y_samples[i].T for i in range(Y_samples.shape[0])], axis=1)  # [N_t, N*N_r]
            W_right = self._ridge_left_solve(Y_cat_t, A_cat_t).T.astype(self.dtype)

            # Convergence check
            Y_hat = np.array([W_left @ X_samples[i] @ W_right for i in range(X_samples.shape[0])], dtype=self.dtype)
            err = np.linalg.norm((Y_samples - Y_hat).reshape(-1))
            denom = np.linalg.norm(Y_samples.reshape(-1)) + 1e-12
            rel_err = float(err / denom)
            if abs(prev_rel_err - rel_err) < tol:
                break
            prev_rel_err = rel_err

        return W_left, W_right
    
    def form_window_input_signal(self, Y_3D_complex, curr_window_weights):
        Y_4D, _ = self._ensure_batch_axis(Y_3D_complex)
        return self._form_window_input_signal_core(Y_4D, curr_window_weights)[:, 0]
    
    def test_train_predict_batch(self, channel_train_input_batch, curr_window_weights):
        Y_4D_org, _ = self._ensure_batch_axis(channel_train_input_batch)
        Y_4D = self._prepare_inputs(Y_4D_org, curr_window_weights)
        S_4D = self._state_transit_core(Y_4D * self.input_scale, reset_state=True)

        T, B = Y_4D.shape[0], Y_4D.shape[1]
        if self.readout_solve_method == "ALS":
            X_samples = self._feature_matrices_from_state_input(S_4D, Y_4D)
            Y_hat = np.array(
                [self.W_out_left @ X_samples[i] @ self.W_out_right for i in range(T * B)],
                dtype=self.dtype,
            )
            curr_channel_pred = Y_hat.reshape(T, B, self.N_r, self.N_t)
            return np.transpose(curr_channel_pred, (1, 2, 3, 0))

        S = self._flatten_features(S_4D, Y_4D)

        curr_channel_pred = self.W_out @ S
        curr_channel_pred = curr_channel_pred.reshape([self.N_r, self.N_t, T, B])
        return np.transpose(curr_channel_pred, (3, 0, 1, 2))
    
    def state_transit_batch(self, Y_4D):
        Y_4D, _ = self._ensure_batch_axis(Y_4D)
        return self._state_transit_core(Y_4D, reset_state=True)
    
    def state_transit(self, Y_3D):
        Y_4D, _ = self._ensure_batch_axis(Y_3D)
        return self._state_transit_core(Y_4D, reset_state=False)[:, 0]

    def complex_tanh(self, Y):
        return np.tanh(np.real(Y)) + 1j * np.tanh(np.imag(Y))
    
    def configure_weights_from_kalman(self, channel_input, curr_window_weights, err_var_input=None):
        Y_4D, _ = self._ensure_batch_axis(channel_input)
        Y_4D = self._prepare_inputs(Y_4D, curr_window_weights)
        if Y_4D.shape[1] != 1:
            Y_seq = Y_4D.reshape(Y_4D.shape[0] * Y_4D.shape[1], self.d_left, self.d_right)
        else:
            Y_seq = Y_4D[:, 0, :, :]

        x_hist = Y_seq.reshape(Y_seq.shape[0], -1).astype(np.complex128)
        if x_hist.shape[0] < 2:
            return

        kf = kalman_filter_pred(lam=self.reg, eps=self.kalman_eps, ar_order=max(1, self.kalman_weight_ar_order))
        p = min(kf.ar_order, x_hist.shape[0] - 1)
        y_hist_tiles = x_hist[:, None, :]
        a_blocks, q_proc = kf._estimate_ar_p_q_joint(y_hist_tiles, p)
        a1 = a_blocks[0]

        q_proc = 0.5 * (q_proc + q_proc.conj().T)
        q_proc += self.kalman_eps * np.eye(q_proc.shape[0], dtype=np.complex128)
        if err_var_input is not None:
            E_4D, _ = self._ensure_batch_axis(err_var_input)
            E_4D = self._prepare_inputs(E_4D, curr_window_weights)
            if E_4D.shape[1] != 1:
                E_seq = E_4D.reshape(E_4D.shape[0] * E_4D.shape[1], self.d_left, self.d_right)
            else:
                E_seq = E_4D[:, 0, :, :]
            e_hist = np.real(E_seq.reshape(E_seq.shape[0], -1))
            r_diag = np.maximum(np.mean(e_hist, axis=0), self.kalman_eps)
        else:
            r_diag = np.maximum(np.real(np.diag(q_proc)), self.kalman_eps)
        r_mat = np.diag(r_diag.astype(np.complex128))

        H = np.eye(a1.shape[0], dtype=np.complex128)
        K = self._compute_steady_kalman_gain(a1, q_proc, r_mat, H)

        self.W_res_kron = (a1 @ (np.eye(a1.shape[0], dtype=np.complex128) - K @ H)).astype(self.dtype)
        self.W_in_kron = (a1 @ K).astype(self.dtype)

        self.use_kalman_bilinear = False
        if self.kalman_project_to_bilinear:
            res_left, res_right_t = self._nearest_kronecker_factors(
                self.W_res_kron.astype(np.complex128),
                left_shape=(self.d_left, self.d_left),
                right_shape=(self.d_right, self.d_right),
            )
            in_left, in_right_t = self._nearest_kronecker_factors(
                self.W_in_kron.astype(np.complex128),
                left_shape=(self.d_left, self.N_in_left),
                right_shape=(self.d_right, self.N_in_right),
            )

            self.W_res_left, self.W_res_right = self._normalize_bilinear_recurrent_radius(
                res_left.astype(self.dtype),
                res_right_t.T.astype(self.dtype),
            )
            self.W_in_left = in_left.astype(self.dtype)
            self.W_in_right = in_right_t.T.astype(self.dtype)
            self.use_kalman_bilinear = True

    def _nearest_kronecker_factors(self, M, left_shape, right_shape):
        # Finds A, B such that M ≈ kron(B, A) in Frobenius norm.
        m, n = left_shape
        p, q = right_shape
        expected_shape = (m * p, n * q)
        if M.shape != expected_shape:
            raise ValueError(
                f"Kronecker projection shape mismatch: got {M.shape}, expected {expected_shape}"
            )

        # Rearrangement for rank-1 SVD: R ≈ vec(A) vec(B)^H when M ≈ kron(B, A)
        R = M.reshape(p, m, q, n).transpose(1, 3, 0, 2).reshape(m * n, p * q)
        U, s, Vh = np.linalg.svd(R, full_matrices=False)
        s0 = float(np.real(s[0])) if s.size > 0 else 0.0
        if not np.isfinite(s0) or s0 <= self.kalman_eps:
            return (
                np.zeros((m, n), dtype=np.complex128),
                np.zeros((p, q), dtype=np.complex128),
            )

        root = np.sqrt(s0)
        vec_a = root * U[:, 0]
        vec_b = root * np.conj(Vh[0, :])
        A = vec_a.reshape(m, n)
        B = vec_b.reshape(p, q)
        return A, B

    def _normalize_bilinear_recurrent_radius(self, W_left, W_right):
        # Keep product spectral radius near configured target for stable recurrent dynamics.
        left_eigs = np.linalg.eigvals(W_left.astype(np.complex128))
        right_eigs = np.linalg.eigvals(W_right.astype(np.complex128))
        rho = float(np.max(np.abs(left_eigs)) * np.max(np.abs(right_eigs)))
        if np.isfinite(rho) and rho > self.kalman_eps:
            scale = np.sqrt(self.spectral_radius / rho)
            W_left = W_left * scale
            W_right = W_right * scale
        return W_left, W_right

    def _compute_steady_kalman_gain(self, F, Q, R, H):
        P = np.eye(F.shape[0], dtype=np.complex128)
        I = np.eye(F.shape[0], dtype=np.complex128)

        for _ in range(max(1, self.kalman_gain_iters)):
            P_pred = F @ P @ F.conj().T + Q
            S = H @ P_pred @ H.conj().T + R
            K = P_pred @ H.conj().T @ np.linalg.pinv(S)
            P = (I - K @ H) @ P_pred

        return K

def _predict_pair_worker(args):
    base_history, err_var_history, rc_config, RB, tx_ant_idx, rx_ant_idx = args

    curr_h_freq_csi_history = base_history[:, :, :, rx_ant_idx, :, ...]
    curr_h_freq_csi_history = curr_h_freq_csi_history[:, :, :, :, :, tx_ant_idx, ...]

    curr_err_var_history = None
    if err_var_history is not None:
        curr_err_var_history = err_var_history[:, :, :, rx_ant_idx, :, ...]
        curr_err_var_history = curr_err_var_history[:, :, :, :, :, tx_ant_idx, ...]

        if curr_h_freq_csi_history.shape != curr_err_var_history.shape:
            sc_diff = curr_h_freq_csi_history.shape[-1] - curr_err_var_history.shape[-1]
            left_pad  = np.repeat(curr_err_var_history[..., :1], sc_diff // 2, axis=-1)
            right_pad = np.repeat(curr_err_var_history[..., -1:], sc_diff - sc_diff // 2, axis=-1)
            curr_err_var_history = np.concatenate([left_pad, curr_err_var_history, right_pad], axis=-1)

            if curr_h_freq_csi_history.shape != curr_err_var_history.shape:
                raise ValueError("curr_h_freq_csi_history and curr_err_var_history must have the same shape")

    twomode_predictor = twomode_wesn_pred(
        rc_config=rc_config,
        num_freq_re=RB,
        num_rx_ant=len(rx_ant_idx),
        num_tx_ant=len(tx_ant_idx),
    )

    tmp = twomode_predictor.predict(
        curr_h_freq_csi_history,
        err_var_history=curr_err_var_history,
    )
    rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
    return rx_idx, tx_idx, tmp
    
def predict_all_links(
    h_freq_csi_history,
    rc_config,
    ns3cfg,
    num_bs_ant=4,
    num_ue_ant=2,
    max_workers=None,
    err_var_csi_history=None,
):
    base_history = np.asarray(h_freq_csi_history)
    err_var_history = None if err_var_csi_history is None else np.asarray(err_var_csi_history)

    _, _, _, _, _, _, _, RB = base_history.shape
    h_freq_csi = np.zeros(base_history[0, ...].shape, dtype=base_history.dtype)

    tasks = []
    for tx_node_idx in range(ns3cfg.num_txue_sel + 1):
        for rx_node_idx in range(ns3cfg.num_rxue_sel + 1):
            if tx_node_idx == 0:
                tx_ant_idx = np.arange(0, num_bs_ant)
            else:
                tx_ant_idx = np.arange(
                    num_bs_ant + (tx_node_idx - 1) * num_ue_ant,
                    num_bs_ant + tx_node_idx * num_ue_ant,
                )

            if rx_node_idx == 0:
                rx_ant_idx = np.arange(0, num_bs_ant)
            else:
                rx_ant_idx = np.arange(
                    num_bs_ant + (rx_node_idx - 1) * num_ue_ant,
                    num_bs_ant + rx_node_idx * num_ue_ant,
                )

            tasks.append((base_history, err_var_history, rc_config, RB, tx_ant_idx, rx_ant_idx))

    if max_workers is None:
        max_workers = min(len(tasks), os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_predict_pair_worker, task): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            rx_idx, tx_idx, tmp = future.result()
            h_freq_csi[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

    return h_freq_csi


def predict_all_links_simple(h_freq_csi_history, rc_config, ns3cfg, num_bs_ant=4, num_ue_ant=2, err_var_csi_history=None):

    T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape
    h_freq_csi = np.zeros(h_freq_csi_history[0, ...].shape, dtype=h_freq_csi_history.dtype)

    for tx_node_idx in range(ns3cfg.num_txue_sel + 1):
        for rx_node_idx in range(ns3cfg.num_rxue_sel + 1):
            if tx_node_idx == 0:
                tx_ant_idx = np.arange(0, num_bs_ant)
            else:
                tx_ant_idx = np.arange(
                    num_bs_ant + (tx_node_idx - 1) * num_ue_ant,
                    num_bs_ant + (tx_node_idx) * num_ue_ant,
                )
            TxAnt = len(tx_ant_idx)

            if rx_node_idx == 0:
                rx_ant_idx = np.arange(0, num_bs_ant)
            else:
                rx_ant_idx = np.arange(
                    num_bs_ant + (rx_node_idx - 1) * num_ue_ant,
                    num_bs_ant + (rx_node_idx) * num_ue_ant,
                )
            RxAnt = len(rx_ant_idx)

            curr_h_freq_csi_history = h_freq_csi_history[:, :, :, rx_ant_idx, :, ...]
            curr_h_freq_csi_history = curr_h_freq_csi_history[:, :, :, :, :, tx_ant_idx, ...]

            curr_h_freq_csi_history = tf.convert_to_tensor(curr_h_freq_csi_history)

            twomode_predictor = twomode_wesn_pred(
                rc_config=rc_config,
                num_freq_re=RB,
                num_rx_ant=RxAnt,
                num_tx_ant=TxAnt,
            )
            rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
            curr_err_var_csi_history = None
            if err_var_csi_history is not None:
                curr_err_var_csi_history = err_var_csi_history[:, :, :, rx_ant_idx, :, ...]
                curr_err_var_csi_history = curr_err_var_csi_history[:, :, :, :, :, tx_ant_idx, ...]
            tmp = np.asarray(twomode_predictor.predict(curr_h_freq_csi_history, err_var_csi_history))
            h_freq_csi[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

    return h_freq_csi