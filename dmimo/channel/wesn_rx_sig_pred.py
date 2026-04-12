import numpy as np


class wesn_rx_sig_pred:
    """
    Standard (non-bilinear) complex WESN for received pilot-signal prediction.

    Input history shape:
      [T, batch, num_rx, num_rx_ant, num_pilot_ofdm_symbols, num_effective_subcarriers]

    Output (next-step prediction) shape:
      [batch, num_rx, num_rx_ant, num_pilot_ofdm_symbols, num_effective_subcarriers]
    """

    def __init__(self, rc_config, num_rx_ant):
        self.rc_config = rc_config
        self.num_rx_ant = num_rx_ant

        self.num_neurons = int(rc_config.num_neurons)
        self.sparsity = float(rc_config.W_tran_sparsity)
        self.spectral_radius = float(rc_config.W_tran_radius)
        self.input_scale = float(rc_config.input_scale)
        self.reg = float(rc_config.regularization)
        self.enable_window = bool(rc_config.enable_window)
        self.window_length = int(rc_config.window_length) if self.enable_window else 1

        if self.enable_window:
            self.input_dim = self.window_length * num_rx_ant
        else:   
            self.input_dim = num_rx_ant

        self.rs = np.random.RandomState(10)

        self._init_weights()

    def _init_weights(self):
        self.w_in = (self.rs.uniform(-1.0, 1.0, (self.num_neurons, self.input_dim)) +
                1j * self.rs.uniform(-1.0, 1.0, (self.num_neurons, self.input_dim)))

        self.w_res = (self.rs.uniform(-1.0, 1.0, (self.num_neurons, self.num_neurons)) +
                 1j * self.rs.uniform(-1.0, 1.0, (self.num_neurons, self.num_neurons)))
        mask = self.rs.uniform(0.0, 1.0, self.w_res.shape) < self.sparsity
        self.w_res[mask] = 0.0 + 0.0j

        eigvals = np.linalg.eigvals(self.w_res)
        radius = np.max(np.abs(eigvals)) if eigvals.size > 0 else 0.0
        if radius > 0:
            self.w_res = self.w_res * (self.spectral_radius / radius)

        return
    
    def complex_tanh(self, y):
        return np.tanh(np.real(y)) + 1j * np.tanh(np.imag(y))
    
    def _make_windowed_pairs(self, seq):
        """Build (X, Y, x_test) for one-step prediction from a sequence [T, B, F]."""

        T, B, F = seq.shape
        if T < 2:
            raise ValueError("Need at least 2 time steps for training/prediction.")

        if not self.enable_window or self.window_length <= 1:
            X = seq[:-1, :, :]
            Y = seq[1:, :, :]
            x_test = seq[-1, :, :]
            return X, Y, x_test

        K = self.window_length
        X = []
        for t in range(T - 1):
            start = max(0, t - K + 1)
            win = seq[start:t + 1, :, :]
            if win.shape[0] < K:
                pad = np.repeat(win[0:1, :, :], K - win.shape[0], axis=0)
                win = np.concatenate([pad, win], axis=0)
            # [K, B, F] -> [B, K*F]
            X.append(np.transpose(win, (1, 0, 2)).reshape(B, K * F))

        start = max(0, T - K)
        test_win = seq[start:T, :, :]
        if test_win.shape[0] < K:
            pad = np.repeat(test_win[0:1, :, :], K - test_win.shape[0], axis=0)
            test_win = np.concatenate([pad, test_win], axis=0)
        x_test = np.transpose(test_win, (1, 0, 2)).reshape(B, K * F)
        return np.asarray(X), seq[1:, :, :], x_test

    def _fit_predict_batched(self, seq):
        """
        Train WESN on sequence seq [T, B, F] and predict next vectors [B, F].
        """

        X, Y, x_test = self._make_windowed_pairs(seq)
        _, B, input_dim = X.shape
        target_dim = Y.shape[-1]

        state = np.zeros((B, self.num_neurons), dtype=np.complex128)
        states = np.zeros((X.shape[0], B, self.num_neurons), dtype=np.complex128)

        for t in range(X.shape[0]):
            u = self.input_scale * X[t, :, :]
            recurrent_term = self.w_res[np.newaxis, :, :] @ state[:,:, np.newaxis]
            input_term = self.w_in[np.newaxis, :, :] @ u[:,:, np.newaxis]
            state = np.squeeze(self.complex_tanh(recurrent_term + input_term))
            states[t, :, :] = state

        # Feature matrix Z across all batches: [num_samples_all_batches, num_neurons + input_dim]
        Z = np.concatenate([states, X], axis=2).reshape(-1, self.num_neurons + input_dim)
        Y_flat = Y.reshape(-1, target_dim)

        # Ridge regression across all batches:
        # W_out = (Z^H Z + reg*I)^-1 Z^H Y_flat
        zhz = Z.conj().T @ Z
        reg_eye = self.reg * np.eye(zhz.shape[0], dtype=zhz.dtype)
        w_out = np.linalg.solve(zhz + reg_eye, Z.conj().T @ Y_flat)

        # Advance all batch states with latest input and predict next outputs
        u_test = self.input_scale * x_test  # [B, input_dim]
        state = self.complex_tanh((state @ self.w_res.T) + (u_test @ self.w_in.T))
        z_test = np.concatenate([state, x_test.reshape(B, input_dim)], axis=1)  # [B, num_neurons + input_dim]
        y_pred = z_test @ w_out  # [B, target_dim]

        return y_pred

    def predict(self, rx_sig_history):
        rx_sig_history = np.asarray(rx_sig_history)
        if rx_sig_history.ndim == 3:
            # Direct mode: [T, batch, features_per_batch] -> [batch, features_per_batch]
            return self._fit_predict_batched(rx_sig_history)
        if rx_sig_history.ndim != 6:
            raise ValueError(
                "rx_sig_history must have shape [T, batch, features_per_batch] or "
                "[T, batch, num_rx, num_rx_ant, num_pilot_ofdm_symbols, num_effective_subcarriers]"
            )

        T, batch_size, num_rx, num_rx_ant, num_pilot_syms, num_freq = rx_sig_history.shape
        if num_rx_ant != self.num_rx_ant:
            raise ValueError("num_rx_ant mismatch in wesn_rx_sig_pred")

        pred = np.zeros((batch_size, num_rx, num_rx_ant, num_pilot_syms, num_freq), dtype=rx_sig_history.dtype)

        for rx_node in range(num_rx):
            seq = rx_sig_history[:, :, rx_node, :, :, :].reshape(T, num_rx_ant, -1)
            seq = np.transpose(seq, (0, 2, 1))  # [T, num_pilot_syms * num_freq, num_rx_ant]
            y_pred = self._fit_predict_batched(seq)
            pred[:, rx_node, :, :, :] = y_pred.reshape(batch_size, num_pilot_syms, num_freq, num_rx_ant).transpose(0, 3, 1, 2)

        return pred
    
def rx_sig_predict_all_links_simple(rx_sig_freq_history, rc_config, ns3cfg, num_bs_ant=4, num_ue_ant=2, err_var_csi_history=None):

    rx_sig_freq = np.zeros(rx_sig_freq_history[0, ...].shape, dtype=rx_sig_freq_history.dtype)

    for rx_node_idx in range(ns3cfg.num_rxue_sel + 1):

        if rx_node_idx == 0:
            rx_ant_idx = np.arange(0, num_bs_ant)
        else:
            rx_ant_idx = np.arange(
                num_bs_ant + (rx_node_idx - 1) * num_ue_ant,
                num_bs_ant + (rx_node_idx) * num_ue_ant,
            )

        curr_rx_sig_freq_history = rx_sig_freq_history[:, :, :, rx_ant_idx, ...]

        _, _, _, num_rx_ant, _, _ = curr_rx_sig_freq_history.shape

        rx_sig_predictor = wesn_rx_sig_pred(
                rc_config=rc_config,
                num_rx_ant=num_rx_ant,
            )
        rx_sig_freq[:, :, rx_ant_idx, :, :] = np.asarray(rx_sig_predictor.predict(curr_rx_sig_freq_history))

    return rx_sig_freq