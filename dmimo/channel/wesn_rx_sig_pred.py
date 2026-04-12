import numpy as np


class wesn_rx_sig_pred:
    """
    Standard (non-bilinear) complex WESN for received pilot-signal prediction.

    Input history shape:
      [T, batch, num_rx, num_rx_ant, num_pilot_ofdm_symbols, num_effective_subcarriers]

    Output (next-step prediction) shape:
      [batch, num_rx, num_rx_ant, num_pilot_ofdm_symbols, num_effective_subcarriers]
    """

    def __init__(self, rc_config, num_rx_ant, num_pilot_ofdm_symbols, num_freq_re):
        self.rc_config = rc_config
        self.num_rx_ant = num_rx_ant
        self.num_pilot_ofdm_symbols = num_pilot_ofdm_symbols
        self.num_freq_re = num_freq_re

        self.num_neurons = int(rc_config.num_neurons)
        self.sparsity = float(rc_config.W_tran_sparsity)
        self.spectral_radius = float(rc_config.W_tran_radius)
        self.input_scale = float(rc_config.input_scale)
        self.reg = float(rc_config.regularization)
        self.enable_window = bool(rc_config.enable_window)
        self.window_length = int(rc_config.window_length) if self.enable_window else 1

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

    def _make_windowed_pairs(self, seq):
        """Build (X, Y, x_test) for one-step prediction from a sequence [T, D]."""

        T, D = seq.shape
        if T < 2:
            raise ValueError("Need at least 2 time steps for training/prediction.")

        if not self.enable_window or self.window_length <= 1:
            X = seq[:-1, :]
            Y = seq[1:, :]
            x_test = seq[-1, :]
            return X, Y, x_test

        K = self.window_length
        X = []
        Y = []
        for t in range(T - 1):
            start = max(0, t - K + 1)
            win = seq[start:t + 1, :]
            if win.shape[0] < K:
                pad = np.repeat(win[0:1, :], K - win.shape[0], axis=0)
                win = np.concatenate([pad, win], axis=0)
            X.append(win.reshape(-1))
            Y.append(seq[t + 1, :])

        start = max(0, T - K)
        test_win = seq[start:T, :]
        if test_win.shape[0] < K:
            pad = np.repeat(test_win[0:1, :], K - test_win.shape[0], axis=0)
            test_win = np.concatenate([pad, test_win], axis=0)

        return np.asarray(X), np.asarray(Y), test_win.reshape(-1)

    def _fit_predict_single(self, seq):
        """
        Train WESN on sequence seq [T, D] and predict next vector [D].
        """

        X, Y, x_test = self._make_windowed_pairs(seq)
        input_dim = X.shape[1]

        state = np.zeros((self.num_neurons,), dtype=np.complex128)

        states = np.zeros((X.shape[0], self.num_neurons), dtype=np.complex128)
        for t in range(X.shape[0]):
            u = self.input_scale * X[t, :]
            state = np.tanh(self.w_in @ u + self.w_res @ state)
            states[t, :] = state

        # Feature matrix Z: [num_samples, num_neurons + input_dim]
        Z = np.concatenate([states, X], axis=1)

        # Ridge regression: W_out maps feature -> target
        # W_out = (Z^H Z + reg*I)^-1 Z^H Y
        zhz = Z.conj().T @ Z
        reg_eye = self.reg * np.eye(zhz.shape[0], dtype=zhz.dtype)
        w_out = np.linalg.solve(zhz + reg_eye, Z.conj().T @ Y)

        # Advance state with latest input and predict next output
        u_test = self.input_scale * x_test
        state = np.tanh(self.w_in @ u_test + self.w_res @ state)
        z_test = np.concatenate([state, x_test], axis=0)
        y_pred = z_test @ w_out

        return y_pred

    def predict(self, rx_sig_history):
        rx_sig_history = np.asarray(rx_sig_history)
        if rx_sig_history.ndim != 6:
            raise ValueError(
                "rx_sig_history must have shape "
                "[T, batch, num_rx, num_rx_ant, num_pilot_ofdm_symbols, num_effective_subcarriers]"
            )

        T, batch_size, num_rx, num_rx_ant, num_pilot_syms, num_freq = rx_sig_history.shape
        if num_rx_ant != self.num_rx_ant:
            raise ValueError("num_rx_ant mismatch in wesn_rx_sig_pred")

        pred = np.zeros((batch_size, num_rx, num_rx_ant, num_pilot_syms, num_freq), dtype=rx_sig_history.dtype)

        for b in range(batch_size):
            for rx_node in range(num_rx):
                seq = rx_sig_history[:, b, rx_node, :, :, :].reshape(T, num_rx_ant, -1)
                seq = seq.transpose(0, 2, 1)
                y_pred = self._fit_predict_single(seq)
                pred[b, rx_node, :, :, :] = y_pred.reshape(num_rx_ant, num_pilot_syms, num_freq)

        return pred
    
def rx_sig_predict_all_links_simple(rx_sig_freq_history, rc_config, ns3cfg, num_bs_ant=4, num_ue_ant=2, err_var_csi_history=None):

    T, _, _, num_rx_ant, num_pilot_syms, num_freq = rx_sig_freq_history.shape
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

        rx_sig_predictor = wesn_rx_sig_pred(
                rc_config=rc_config,
                num_rx_ant=num_rx_ant,
            )
        curr_err_var_csi_history = None
        if err_var_csi_history is not None:
            curr_err_var_csi_history = err_var_csi_history[:, :, :, rx_ant_idx, :, ...]
        tmp = np.asarray(rx_sig_predictor.predict(curr_rx_sig_freq_history, curr_err_var_csi_history))
        rx_sig_freq[:, :, rx_node_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

    return rx_sig_freq