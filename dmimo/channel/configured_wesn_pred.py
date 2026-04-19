import numpy as np

from dmimo.channel.kalman_filter_pred import kalman_filter_pred


class configured_wesn_pred:
    """Configured WESN with non-bilinear dynamics.

    State update:
        s(t) = sigma(W_in y(t) + W_res s(t-1))
    Output:
        y_hat(t) = W_out [s(t); y(t)]
    """

    def __init__(self, rc_config, num_freq_re, num_rx_ant, num_tx_ant, type=np.complex64):
        self.rc_config = rc_config
        self.dtype = type

        self.num_freq_re = num_freq_re
        self.N_r = num_rx_ant
        self.N_t = num_tx_ant

        self.sparsity = float(rc_config.W_tran_sparsity)
        self.spectral_radius = float(rc_config.W_tran_radius)
        self.input_scale = float(rc_config.input_scale)
        self.window_length = int(rc_config.window_length)
        self.reg = float(rc_config.regularization)
        self.enable_window = bool(rc_config.enable_window)
        self.enable_kalman_weight_config = bool(getattr(rc_config, "enable_kalman_weight_config", False))
        self.kalman_weight_ar_order = int(rc_config.window_length)
        self.kalman_gain_iters = int(getattr(rc_config, "kalman_gain_iters", 100))
        self.kalman_eps = float(getattr(rc_config, "kalman_eps", 1e-8))

        self.input_dim = int(self.N_r * (self.N_t * self.window_length if self.enable_window else self.N_t))
        # configured mode uses state dimension aligned with input features for Kalman mapping
        self.state_dim = int(self.input_dim)
        self.output_dim = int(self.N_r * self.N_t)
        self.feature_dim = int(self.state_dim + self.input_dim)

        self.RS = np.random.RandomState(10)
        self.init_weights()

    def init_weights(self):
        self.W_res = self._sparse_mat(self.state_dim)
        self.W_in = (
            self.RS.uniform(-1.0, 1.0, (self.state_dim, self.input_dim))
            + 1j * self.RS.uniform(-1.0, 1.0, (self.state_dim, self.input_dim))
        ).astype(self.dtype)
        self.W_out = self.RS.randn(self.output_dim, self.feature_dim).astype(self.dtype)
        self.S_0 = np.zeros((self.state_dim,), dtype=self.dtype)

    def _sparse_mat(self, m):
        W = (
            self.RS.uniform(-1.0, 1.0, (m, m))
            + 1j * self.RS.uniform(-1.0, 1.0, (m, m))
        )
        mask = self.RS.uniform(0.0, 1.0, W.shape) < self.sparsity
        W[mask] = 0.0 + 0.0j
        eigvals = np.linalg.eigvals(W)
        radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        if np.isfinite(radius) and radius > 1e-12:
            W = W * (self.spectral_radius / radius)
        return W.astype(self.dtype)

    def _activation(self, x):
        return np.tanh(np.real(x)) + 1j * np.tanh(np.imag(x))

    def _window_sequence(self, seq):
        # seq: [T, B, N_r, N_t] -> [T, B, input_dim]
        T, B, N_r, N_t = seq.shape
        if not self.enable_window or self.window_length <= 1:
            return seq.reshape(T, B, N_r * N_t)

        K = self.window_length
        out = np.zeros((T, B, N_r * N_t * K), dtype=self.dtype)
        for t in range(T):
            start = max(0, t - K + 1)
            win = seq[start : t + 1]
            if win.shape[0] < K:
                pad = np.repeat(win[0:1], K - win.shape[0], axis=0)
                win = np.concatenate([pad, win], axis=0)
            out[t] = np.transpose(win, (1, 0, 2, 3)).reshape(B, -1)
        return out

    def _state_rollout(self, u_seq):
        # u_seq: [T, B, input_dim]
        T, B, _ = u_seq.shape
        state = np.repeat(self.S_0[None, :], B, axis=0)
        states = np.zeros((T, B, self.state_dim), dtype=self.dtype)
        for t in range(T):
            u_t = self.input_scale * u_seq[t]
            state = self._activation((u_t @ self.W_in.T) + (state @ self.W_res.T))
            states[t] = state
        self.S_0 = state[0].copy()
        return states

    def _fit_readout(self, states, inputs, targets):
        # states/inputs: [T,B,*], targets: [T,B,output_dim]
        feat = np.concatenate([states, inputs], axis=-1).reshape(-1, self.feature_dim)
        y = targets.reshape(-1, self.output_dim)
        gram = feat.conj().T @ feat + self.reg * np.eye(self.feature_dim, dtype=np.complex128)
        self.W_out = (np.linalg.pinv(gram) @ feat.conj().T @ y).T.astype(self.dtype)

    def _compute_steady_kalman_gain(self, F, Q, R, H):
        P = np.eye(F.shape[0], dtype=np.complex128)
        I = np.eye(F.shape[0], dtype=np.complex128)
        for _ in range(max(1, self.kalman_gain_iters)):
            P_pred = F @ P @ F.conj().T + Q
            S = H @ P_pred @ H.conj().T + R
            K = P_pred @ H.conj().T @ np.linalg.pinv(S)
            P = (I - K @ H) @ P_pred
        return K

    def configure_weights_from_kalman(self, u_seq, err_var_seq=None):
        # u_seq: [T,B,input_dim]
        x_hist = u_seq.reshape(-1, u_seq.shape[-1]).astype(np.complex128)
        if x_hist.shape[0] < 2:
            return

        kf = kalman_filter_pred(lam=self.reg, eps=self.kalman_eps, ar_order=max(1, self.kalman_weight_ar_order))
        p = min(kf.ar_order, x_hist.shape[0] - 1)
        a_blocks, q_proc = kf._estimate_ar_p_q_joint(x_hist[:, None, :], p)
        a1 = a_blocks[0]

        q_proc = 0.5 * (q_proc + q_proc.conj().T)
        q_proc += self.kalman_eps * np.eye(q_proc.shape[0], dtype=np.complex128)

        if err_var_seq is not None:
            e_hist = np.real(err_var_seq.reshape(-1, err_var_seq.shape[-1]))
            r_diag = np.maximum(np.mean(e_hist, axis=0), self.kalman_eps)
        else:
            r_diag = np.maximum(np.real(np.diag(q_proc)), self.kalman_eps)

        H = np.eye(a1.shape[0], dtype=np.complex128)
        K = self._compute_steady_kalman_gain(a1, q_proc, np.diag(r_diag.astype(np.complex128)), H)

        W_res = a1 @ (np.eye(a1.shape[0], dtype=np.complex128) - K @ H)
        W_in = a1 @ K

        # resize if needed (should normally match input_dim/state_dim)
        d = min(self.state_dim, W_res.shape[0])
        self.W_res = np.zeros((self.state_dim, self.state_dim), dtype=self.dtype)
        self.W_res[:d, :d] = W_res[:d, :d]

        self.W_in = np.zeros((self.state_dim, self.input_dim), dtype=self.dtype)
        self.W_in[:d, : min(self.input_dim, W_in.shape[1])] = W_in[:d, : min(self.input_dim, W_in.shape[1])]

    def predict(self, h_freq_csi_history, err_var_history=None):
        h = np.asarray(h_freq_csi_history)
        e = None if err_var_history is None else np.asarray(err_var_history)
        if h.ndim != 8:
            raise ValueError("Configured WESN expects 8D CSI history tensor.")

        # align to [T, batch, tx_node, rx_ant, rx_node, tx_ant, freq, ofdm]
        h = h.transpose([0, 1, 2, 3, 4, 5, 7, 6])
        if e is not None:
            e = e.transpose([0, 1, 2, 3, 4, 5, 7, 6])

        num_freq = h.shape[6]
        num_ofdm = h.shape[7]

        train_in = h[:-1]
        train_gt = h[1:]
        err_in = None if e is None else e[:-1]

        pred = np.zeros(h[0].shape, dtype=self.dtype)

        y_in = train_in[:, 0, 0, :, 0, :, :, :]  # [T,Nr,Nt,F,O]
        y_gt = train_gt[:, 0, 0, :, 0, :, :, :]

        T = y_in.shape[0]
        B = num_freq * num_ofdm

        y_in_batch = np.transpose(y_in, (0, 3, 4, 1, 2)).reshape(T, B, self.N_r, self.N_t)
        y_gt_batch = np.transpose(y_gt, (0, 3, 4, 1, 2)).reshape(T, B, self.N_r, self.N_t)

        u_seq = self._window_sequence(y_in_batch)
        y_seq = y_gt_batch.reshape(T, B, self.output_dim)

        if self.enable_kalman_weight_config:
            err_seq = None
            if err_in is not None:
                e_in = err_in[:, 0, 0, :, 0, :, :, :]
                e_batch = np.transpose(e_in, (0, 3, 4, 1, 2)).reshape(T, B, self.N_r, self.N_t)
                err_seq = self._window_sequence(e_batch)
            self.configure_weights_from_kalman(u_seq, err_var_seq=err_seq)

        states = self._state_rollout(u_seq)
        self._fit_readout(states, u_seq, y_seq)

        test_in = train_gt[:, 0, 0, :, 0, :, :, :]
        test_batch = np.transpose(test_in, (0, 3, 4, 1, 2)).reshape(T, B, self.N_r, self.N_t)
        u_test = self._window_sequence(test_batch)
        s_test = self._state_rollout(u_test)
        feat_test = np.concatenate([s_test, u_test], axis=-1)
        y_hat = (feat_test.reshape(-1, self.feature_dim) @ self.W_out.T).reshape(T, B, self.output_dim)

        last = y_hat[-1].reshape(num_freq, num_ofdm, self.N_r, self.N_t)
        last = np.transpose(last, (2, 3, 0, 1))
        pred[:, 0, :, 0, :, :, :] = last

        return pred.transpose([0, 1, 2, 3, 4, 6, 5])


def _freeze_predictor_weights(predictor):
    predictor.enable_kalman_weight_config = False


def build_configured_predictors_simple(
    h_freq_csi_history,
    rc_config,
    ns3cfg,
    num_bs_ant=4,
    num_ue_ant=2,
    err_var_csi_history=None,
):
    _, _, _, _, _, _, _, num_rb = h_freq_csi_history.shape
    configured_predictors = {}

    for tx_node_idx in range(ns3cfg.num_txue_sel + 1):
        for rx_node_idx in range(ns3cfg.num_rxue_sel + 1):
            tx_ant_idx = np.arange(0, num_bs_ant) if tx_node_idx == 0 else np.arange(
                num_bs_ant + (tx_node_idx - 1) * num_ue_ant,
                num_bs_ant + tx_node_idx * num_ue_ant,
            )
            rx_ant_idx = np.arange(0, num_bs_ant) if rx_node_idx == 0 else np.arange(
                num_bs_ant + (rx_node_idx - 1) * num_ue_ant,
                num_bs_ant + rx_node_idx * num_ue_ant,
            )

            curr_h = h_freq_csi_history[:, :, :, rx_ant_idx, :, ...]
            curr_h = curr_h[:, :, :, :, :, tx_ant_idx, ...]

            curr_e = None
            if err_var_csi_history is not None:
                curr_e = err_var_csi_history[:, :, :, rx_ant_idx, :, ...]
                curr_e = curr_e[:, :, :, :, :, tx_ant_idx, ...]

            predictor = configured_wesn_pred(
                rc_config=rc_config,
                num_freq_re=num_rb,
                num_rx_ant=len(rx_ant_idx),
                num_tx_ant=len(tx_ant_idx),
            )
            _ = np.asarray(predictor.predict(curr_h, curr_e))
            _freeze_predictor_weights(predictor)
            configured_predictors[(tx_node_idx, rx_node_idx)] = predictor

    return configured_predictors


def predict_all_links_with_configured_simple(
    h_freq_csi_history,
    configured_predictors,
    ns3cfg,
    num_bs_ant=4,
    num_ue_ant=2,
    err_var_csi_history=None,
):
    h_freq_csi = np.zeros(h_freq_csi_history[0, ...].shape, dtype=h_freq_csi_history.dtype)

    for tx_node_idx in range(ns3cfg.num_txue_sel + 1):
        for rx_node_idx in range(ns3cfg.num_rxue_sel + 1):
            predictor = configured_predictors[(tx_node_idx, rx_node_idx)]

            tx_ant_idx = np.arange(0, num_bs_ant) if tx_node_idx == 0 else np.arange(
                num_bs_ant + (tx_node_idx - 1) * num_ue_ant,
                num_bs_ant + tx_node_idx * num_ue_ant,
            )
            rx_ant_idx = np.arange(0, num_bs_ant) if rx_node_idx == 0 else np.arange(
                num_bs_ant + (rx_node_idx - 1) * num_ue_ant,
                num_bs_ant + rx_node_idx * num_ue_ant,
            )

            curr_h = h_freq_csi_history[:, :, :, rx_ant_idx, :, ...]
            curr_h = curr_h[:, :, :, :, :, tx_ant_idx, ...]

            curr_e = None
            if err_var_csi_history is not None:
                curr_e = err_var_csi_history[:, :, :, rx_ant_idx, :, ...]
                curr_e = curr_e[:, :, :, :, :, tx_ant_idx, ...]

            tmp = np.asarray(predictor.predict(curr_h, curr_e))
            rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
            h_freq_csi[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

    return h_freq_csi