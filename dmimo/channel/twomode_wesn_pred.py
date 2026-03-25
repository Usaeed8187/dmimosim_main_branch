import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

        h_freq_csi_predicted = self.pred(h_freq_csi_history)

        return h_freq_csi_predicted
    
    def pred(self, h_freq_csi_history):
        
        h_freq_csi_history = np.asarray(h_freq_csi_history)
        if h_freq_csi_history.ndim == 8:
            h_freq_csi_history = h_freq_csi_history.transpose([0,1,2,3,4,5,7,6])
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
                # Flatten (freq_re, ofdm_sym) into a single axis and build one long sequence:
                # [T, N_r, N_t, F, O] -> [F, O, T, N_r, N_t] -> [F*O*T, N_r, N_t]
                Y_in_all = channel_train_input[:, 0, tx_node, :, rx_node, :, :, :]
                Y_out_all = channel_train_gt[:, 0, tx_node, :, rx_node, :, :, :]

                T_train = Y_in_all.shape[0]
                B_fo = num_freq_res * num_ofdm_syms

                Y_in_seq = np.transpose(Y_in_all, (3, 4, 0, 1, 2)).reshape(
                    B_fo * T_train, self.N_r, self.N_t
                )
                Y_out_seq = np.transpose(Y_out_all, (3, 4, 0, 1, 2)).reshape(
                    B_fo * T_train, self.N_r, self.N_t
                )

                S_all, Y_all = self.build_S_Y(Y_in_seq, Y_out_seq, curr_window_weights=None)

                # --------- (B) SINGLE READOUT SOLVE (shared across RBs) ----------
                # Prefer ridge for stability:
                G = self.reg_p_inv(S_all)               # (sum_T, F)  :=  S_all^H (S_all S_all^H + λI)^{-1}
                self.W_out = Y_all @ G                  # (N_r*N_t, F)

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

        Y_4D_window = np.zeros((T, B, N_r, L * N_t), dtype=self.dtype)

        for k in range(T):
            blocks = []
            for ell in range(L):
                t = k - ell
                if t >= 0:
                    blocks.append(Y_4D_complex[t])       # [B, N_r, N_t]
                else:
                    blocks.append(np.zeros((B, N_r, N_t), dtype=self.dtype))
            Y_4D_window[k] = np.concatenate(blocks, axis=-1)

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
            recurrent_term = (self.W_res_left @ S_3D) @ self.W_res_right
            input_term = (self.W_in_left @ Y_4D[t]) @ self.W_in_right
            S_3D = self.complex_tanh(recurrent_term + input_term)
            S_4D.append(S_3D)

        S_4D = np.stack(S_4D, axis=0)

        self.S_0 = S_3D[0]

        return S_4D

    def _flatten_features(self, S_4D, Y_4D):
        S_aug_4D = np.concatenate([S_4D, Y_4D], axis=-1)
        T, B = S_aug_4D.shape[:2]
        return S_aug_4D.reshape(T * B, -1).T
    
    def _flatten_targets(self, Y_4D):
        T, B = Y_4D.shape[:2]
        return Y_4D.reshape(T * B, -1).T
    
    def form_window_input_signal(self, Y_3D_complex, curr_window_weights):
        Y_4D, _ = self._ensure_batch_axis(Y_3D_complex)
        return self._form_window_input_signal_core(Y_4D, curr_window_weights)[:, 0]
    
    def test_train_predict_batch(self, channel_train_input_batch, curr_window_weights):
        Y_4D_org, _ = self._ensure_batch_axis(channel_train_input_batch)
        Y_4D = self._prepare_inputs(Y_4D_org, curr_window_weights)
        S_4D = self._state_transit_core(Y_4D * self.input_scale, reset_state=True)
        S = self._flatten_features(S_4D, Y_4D)

        T, B = Y_4D.shape[0], Y_4D.shape[1]
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
    
def predict_all_links(h_freq_csi_history, rc_config, ns3cfg, num_bs_ant=4, num_ue_ant=2, max_workers=None):

    base_history = np.asarray(h_freq_csi_history)
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
                    num_bs_ant + (tx_node_idx) * num_ue_ant,
                )
            if rx_node_idx == 0:
                rx_ant_idx = np.arange(0, num_bs_ant)
            else:
                rx_ant_idx = np.arange(
                    num_bs_ant + (rx_node_idx - 1) * num_ue_ant,
                    num_bs_ant + (rx_node_idx) * num_ue_ant,
                )

            tasks.append((base_history, rc_config, RB, tx_ant_idx, rx_ant_idx))

    if max_workers is None:
        max_workers = min(len(tasks), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_predict_pair_worker, task): task for task in tasks}
        for future in as_completed(future_to_task):
            rx_idx, tx_idx, tmp = future.result()
            h_freq_csi[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

    return h_freq_csi


def predict_all_links_simple(h_freq_csi_history, rc_config, ns3cfg, num_bs_ant=4, num_ue_ant=2):

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
            tmp = np.asarray(twomode_predictor.predict(curr_h_freq_csi_history))
            h_freq_csi[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

    return h_freq_csi