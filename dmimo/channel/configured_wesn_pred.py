import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import lsqr
from scipy.sparse import coo_matrix

from dmimo.channel.kalman_filter_pred import kalman_filter_pred


def apply_complex_activation(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "identity":
        return x
    if activation == "tanh":
        return np.tanh(np.real(x)) + 1j * np.tanh(np.imag(x))
    if activation == "relu":
        return np.maximum(np.real(x), 0.0) + 1j * np.maximum(np.imag(x), 0.0)
    raise ValueError(f"Unsupported activation: {activation}")


def build_augmented_obs_matrix(d: int, p: int) -> np.ndarray:
    h_mat = np.zeros((d, p * d), dtype=np.complex128)
    h_mat[:, :d] = np.eye(d, dtype=np.complex128)
    return h_mat


def solve_riccati_steady_state_complex(
    f_aug: np.ndarray,
    q_aug: np.ndarray,
    h_mat: np.ndarray,
    r_mat: np.ndarray,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    p_minus = np.eye(f_aug.shape[0], dtype=np.complex128)

    for _ in range(max_iter):
        s = h_mat @ p_minus @ h_mat.conj().T + r_mat
        k = p_minus @ h_mat.conj().T @ np.linalg.pinv(s)
        p_plus = p_minus - k @ h_mat @ p_minus
        p_next = f_aug @ p_plus @ f_aug.conj().T + q_aug
        p_next = 0.5 * (p_next + p_next.conj().T)

        if np.linalg.norm(p_next - p_minus, ord="fro") < tol:
            p_minus = p_next
            break
        p_minus = p_next

    s = h_mat @ p_minus @ h_mat.conj().T + r_mat
    k_ss = p_minus @ h_mat.conj().T @ np.linalg.pinv(s)
    return p_minus, k_ss


def steady_state_predictor_transfer_samples_from_kalman(
    f_aug: np.ndarray,
    k_ss: np.ndarray,
    d: int,
    num_freqs: int,
) -> np.ndarray:
    pd = f_aug.shape[0]
    h_mat = np.zeros((d, pd), dtype=np.complex128)
    h_mat[:, :d] = np.eye(d, dtype=np.complex128)
    c_out = h_mat.copy()
    a_p = f_aug - f_aug @ k_ss @ h_mat
    b_p = f_aug @ k_ss
    omegas = np.linspace(0.0, np.pi, num_freqs, endpoint=True)
    eye = np.eye(pd, dtype=np.complex128)
    hs = np.zeros((num_freqs, d, d), dtype=np.complex128)
    for i, w in enumerate(omegas):
        zinv = np.exp(-1j * w)
        hs[i] = c_out @ np.linalg.pinv(eye - a_p * zinv) @ b_p
    return hs


def vectorize_transfer_samples(h_samps: np.ndarray) -> np.ndarray:
    return h_samps.transpose(0, 2, 1).reshape(-1)


def estimate_empirical_complex_covariance(v: np.ndarray, compute_kvv: bool = False):
    mean_v = np.mean(v, axis=1, keepdims=True)
    vc = v - mean_v
    kvv = None
    if compute_kvv:
        kvv = (vc @ vc.conj().T) / max(v.shape[1], 1)
        kvv = 0.5 * (kvv + kvv.conj().T)
    return mean_v, vc, kvv


def q_column_to_frequency_matrix(q_col: np.ndarray, num_freqs: int, value_dim: int) -> np.ndarray:
    return q_col.reshape(num_freqs, value_dim)


def fit_shared_denominator_vector_rational(
    q_col: np.ndarray, num_freqs: int, value_dim: int, degree: int, omegas: np.ndarray
):
    qf = q_column_to_frequency_matrix(q_col, num_freqs, value_dim)
    zinv = np.exp(-1j * omegas)
    z = np.stack([zinv ** k for k in range(1, degree + 1)], axis=1)
    num_rows = num_freqs * value_dim
    q_col = qf.reshape(num_rows)
    z_rep = np.repeat(z, value_dim, axis=0)

    # Build sparse LS system:
    # [diag(q_col) @ z_rep, -blockdiag(z, ..., z)] @ theta = -q_col
    rows_dense = np.repeat(np.arange(num_rows), degree)
    cols_dense = np.tile(np.arange(degree), num_rows)
    vals_dense = (q_col[:, None] * z_rep).reshape(-1)

    r_ids = np.tile(np.arange(value_dim), num_freqs)
    deg_ids = np.tile(np.arange(degree), num_rows)
    cols_block = degree + np.repeat(r_ids, degree) * degree + deg_ids
    vals_block = (-z_rep).reshape(-1)

    rows = np.concatenate([rows_dense, rows_dense])
    cols = np.concatenate([cols_dense, cols_block])
    vals = np.concatenate([vals_dense, vals_block])

    a_ls = coo_matrix(
        (vals, (rows, cols)),
        shape=(num_rows, degree + value_dim * degree),
        dtype=np.complex128,
    ).tocsr()

    y_ls = -q_col
    theta = lsqr(a_ls, y_ls, atol=1e-8, btol=1e-8, iter_lim=2000)[0]
    a_den = theta[:degree]
    b_num = theta[degree:].reshape(value_dim, degree)
    return {"a_den": a_den, "b_num": b_num}


def denominator_coeffs_to_poles(a_den: np.ndarray) -> np.ndarray:
    coeff_desc = np.concatenate([a_den[::-1], np.array([1.0 + 0.0j])])
    roots_x = np.roots(coeff_desc)
    return np.where(np.abs(roots_x) < 1e-12, 0.0 + 0.0j, 1.0 / roots_x).astype(np.complex128)


def stabilize_poles(poles: np.ndarray, max_radius: float = 0.98) -> np.ndarray:
    out = poles.copy().astype(np.complex128)
    mags = np.abs(out)
    scales = np.where(mags >= max_radius, max_radius / (mags + 1e-12), 1.0)
    return out * scales


def decompose_rp_fit_into_first_order(
    fit: dict, q_col: np.ndarray, num_freqs: int, value_dim: int, omegas: np.ndarray
):
    qf = q_column_to_frequency_matrix(q_col, num_freqs, value_dim)
    poles = stabilize_poles(denominator_coeffs_to_poles(fit["a_den"]))
    zinv = np.exp(-1j * omegas)
    phi = 1.0 / (1.0 - zinv[:, None] * poles[None, :])
    residues_t, *_ = np.linalg.lstsq(phi, qf, rcond=None)
    residues = residues_t.T
    return poles, residues

class configured_wesn_pred:
    """Configured WESN"""
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
        self.kalman_eps = float(getattr(rc_config, "kalman_eps", 1e-8))

        self.esn_m = int(getattr(rc_config, "esn_m", 4))
        self.esn_k = int(getattr(rc_config, "esn_k", 4))
        self.esn_num_freqs = int(getattr(rc_config, "esn_num_freqs", 64))
        self.esn_activation = str(getattr(rc_config, "esn_activation", "tanh"))
        self.esn_ls_reg = float(getattr(rc_config, "esn_ls_reg", self.reg))
        self.esn_diagnostics = bool(getattr(rc_config, "esn_diagnostics", False))
        self.enable_skip_connections = bool(getattr(rc_config, "enable_skip_connections", True))

        self.input_dim = int(self.N_r * (self.N_t * self.window_length if self.enable_window else self.N_t))
        self.state_dim = int(max(self.input_dim, self.esn_m * self.esn_k * self.N_r * self.N_t))
        self.output_dim = int(self.N_r * self.N_t)
        self.feature_dim = int(self.state_dim + self.input_dim if self.enable_skip_connections else self.state_dim)

        self.RS = np.random.RandomState(10)
        self.is_frozen = False
        self.init_weights()

    def init_weights(self):
        self.W_res = self._sparse_mat(self.state_dim)
        self.W_in = (
            self.RS.uniform(-1.0, 1.0, (self.state_dim, self.input_dim))
            + 1j * self.RS.uniform(-1.0, 1.0, (self.state_dim, self.input_dim))
        ).astype(self.dtype)
        self.W_out = self.RS.randn(self.output_dim, self.feature_dim).astype(self.dtype)

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

    def _window_sequence(self, seq):
        # seq: [T, B, N_r, N_t] -> [T, B, input_dim]
        T, B, N_r, N_t = seq.shape
        if not self.enable_window or self.window_length <= 1:
            return seq.reshape(T, B, N_r * N_t)

        K = self.window_length
        out = np.zeros((T, B, N_r * N_t * K), dtype=self.dtype)
        for t in range(T):
            start = max(0, t - K + 1)
            win = seq[start: t + 1]
            if win.shape[0] < K:
                pad = np.repeat(win[0:1], K - win.shape[0], axis=0)
                win = np.concatenate([pad, win], axis=0)
            out[t] = np.transpose(win, (1, 0, 2, 3)).reshape(B, -1)
        return out

    def _activation(self, x):
        return apply_complex_activation(x, self.esn_activation)

    def _state_rollout(self, u_seq):
        # u_seq: [T, B, input_dim]
        T, B, _ = u_seq.shape
        state = np.zeros((B, self.state_dim), dtype=self.dtype)
        states = np.zeros((T, B, self.state_dim), dtype=self.dtype)
        for t in range(T):
            u_t = self.input_scale * u_seq[t]
            state = self._activation((u_t @ self.W_in.T) + (state @ self.W_res.T))
            states[t] = state
        return states.astype(self.dtype)

    def _fit_readout(self, states, inputs, targets):
        if self.enable_skip_connections:
            feat = np.concatenate([states, inputs], axis=-1).reshape(-1, self.feature_dim)
        else:
            feat = states.reshape(-1, self.feature_dim)
        y = targets.reshape(-1, self.output_dim)
        gram = feat.conj().T @ feat + self.esn_ls_reg * np.eye(self.feature_dim, dtype=self.dtype)
        self.W_out = (np.linalg.pinv(gram) @ feat.conj().T @ y).T.astype(self.dtype)


    def _configure_weights_from_kalman_stats(self, u_seq, err_var_seq=None):
        # u_seq: [T,B,input_dim]
        t_dec, _, d = u_seq.shape
        if t_dec <= max(2, self.window_length):
            return

        p_eff = min(max(1, self.kalman_weight_ar_order), max(1, self.window_length))
        history_len = min(max(3, self.window_length + 1), t_dec)
        omegas = np.linspace(0.0, np.pi, self.esn_num_freqs, endpoint=True)
        kf_helper = kalman_filter_pred(ar_order=max(1, self.kalman_weight_ar_order))

        if err_var_seq is not None:
            r_diag = np.maximum(np.mean(np.real(err_var_seq.reshape(-1, d)), axis=0), self.kalman_eps)
        else:
            r_diag = np.ones((d,), dtype=np.float64) * self.kalman_eps

        v_list = []
        max_start = max(1, t_dec - history_len)
        for s in range(max_start):
            y_hist_chunk = u_seq[s: s + history_len].astype(np.complex64)
            a_blocks, q_proc = kf_helper._estimate_ar_p_q_joint(y_hist_chunk, p_eff)
            a_blocks = [a.conj() for a in a_blocks]
            f_aug, q_aug = kf_helper._build_augmented_system(a_blocks, q_proc)
            h_mat = build_augmented_obs_matrix(d, p_eff)
            r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex64))
            _, k_ss = solve_riccati_steady_state_complex(f_aug, q_aug, h_mat, r_mat)
            h_samps = steady_state_predictor_transfer_samples_from_kalman(
                f_aug,
                k_ss,
                d=d,
                num_freqs=self.esn_num_freqs,
            )
            v_list.append(vectorize_transfer_samples(h_samps))

        if len(v_list) == 0:
            return

        v = np.stack(v_list, axis=1)
        _, vc, kvv = estimate_empirical_complex_covariance(v)

        m = self.esn_m

        if vc.shape[0] > vc.shape[1] and m <= vc.shape[1]:
            u, svals, _ = np.linalg.svd(vc, full_matrices=False)
            evals = (svals ** 2) / max(vc.shape[1], 1)
            q_eig = u[:, :m]
        else:
            # if m < kvv.shape[0]:
            #     evals, q_eig = eigsh(kvv, k=m, which="LA")
            # else:
            #     evals, evecs = np.linalg.eigh(kvv)
            #     q_eig = evecs
            raise Exception("Not enough time-series data to get m eigenvectors. Consider reducing esn_m or reducing window size.")
        evals_real_sorted = np.sort(np.real(evals))[::-1]
        
        idx = np.argsort(np.real(evals))[::-1][:m]
        q_eig = q_eig[:, idx]

        value_dim = d * d
        poles_all = np.zeros((m, self.esn_k), dtype=np.complex64)
        residues_all = np.zeros((m, value_dim, self.esn_k), dtype=np.complex64)

        for j in range(m):
            q_col = q_eig[:, j]
            fit = fit_shared_denominator_vector_rational(
                q_col=q_col,
                num_freqs=self.esn_num_freqs,
                value_dim=value_dim,
                degree=self.esn_k,
                omegas=omegas,
            )
            poles, residues = decompose_rp_fit_into_first_order(
                fit=fit,
                q_col=q_col,
                num_freqs=self.esn_num_freqs,
                value_dim=value_dim,
                omegas=omegas,
            )
            poles_all[j] = poles
            residues_all[j] = residues

        pole_mags = np.abs(poles_all)
        pole_scales = np.where(
            pole_mags > float(self.spectral_radius),
            float(self.spectral_radius) / (pole_mags + 1e-12),
            1.0,
        )
        poles_all = poles_all * pole_scales

        w_res = np.tile(poles_all.reshape(-1), int(np.ceil(self.state_dim / (m * self.esn_k))))[: self.state_dim]
        self.W_res = np.diag(w_res.astype(self.dtype))

        w_in_small = np.transpose(residues_all, (0, 2, 1)).reshape(m * self.esn_k * d, d, order="F")
        w_in = np.zeros((self.state_dim, self.input_dim), dtype=np.complex64)
        row_lim = min(self.state_dim, w_in_small.shape[0])
        col_lim = min(self.input_dim, w_in_small.shape[1])
        w_in[:row_lim, :col_lim] = w_in_small[:row_lim, :col_lim]
        self.W_in = (self.input_scale * w_in).astype(self.dtype)

        if self.esn_diagnostics:
            total_eval = float(np.sum(np.maximum(evals_real_sorted, 0.0)))
            cum_energy = (
                np.cumsum(np.maximum(evals_real_sorted, 0.0)) / max(total_eval, 1e-15)
                if evals_real_sorted.size > 0
                else np.asarray([], dtype=np.float64)
            )
            suggested_m = {}
            for thr in (0.90, 0.95, 0.99):
                idx_thr = int(np.searchsorted(cum_energy, thr, side="left")) + 1 if cum_energy.size > 0 else 0
                suggested_m[thr] = min(idx_thr, int(evals_real_sorted.size))
            print(
                f"[configured_wesn] m={m}, k={self.esn_k}, num_freqs={self.esn_num_freqs}, "
                f"pole|.| p50={float(np.quantile(np.abs(poles_all), 0.5)):.3f}, "
                f"p90={float(np.quantile(np.abs(poles_all), 0.9)):.3f}"
            )
            print(
                f"[configured_wesn] suggested esn_m by explained-energy thresholds: "
                f"90%->{suggested_m[0.90]}, 95%->{suggested_m[0.95]}, 99%->{suggested_m[0.99]}"
            )

    def fit_offline(self, h_freq_csi_history, err_var_history=None):
        h = np.asarray(h_freq_csi_history)
        e = None if err_var_history is None else np.asarray(err_var_history)
        if h.ndim != 8:
            raise ValueError("Configured WESN expects 8D CSI history tensor.")

        num_freq_res = h.shape[-1]
        if e is not None and e.shape[-1] != num_freq_res:
            pad_top = (num_freq_res - e.shape[-1]) // 2
            pad_bottom = (num_freq_res - e.shape[-1]) - pad_top

            top_rep = np.repeat(e[..., 0:1], pad_top, axis=7)
            bottom_rep  = np.repeat(e[..., -1:],  pad_bottom, axis=7)

            e = np.concatenate([top_rep, e, bottom_rep], axis=7)

        # align to [T, batch, rx_node, rx_ant, tx_node, tx_ant, freq, ofdm]
        h = h.transpose([0, 1, 2, 3, 4, 5, 7, 6])
        if e is not None:
            e = e.transpose([0, 1, 2, 3, 4, 5, 7, 6])

        train_in = h[:-1]
        train_gt = h[1:]
        err_in = None if e is None else e[:-1]

        if train_in.shape[0] < 2:
            return

        y_in = train_in[:, 0, 0, :, 0, :, :, :]
        y_gt = train_gt[:, 0, 0, :, 0, :, :, :]

        T = y_in.shape[0]
        B = y_in.shape[3] * y_in.shape[4]

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
            self._configure_weights_from_kalman_stats(u_seq, err_var_seq=err_seq)

        # states = self._state_rollout(u_seq)
        # self._fit_readout(states, u_seq, y_seq)
        self.is_frozen = True

    def predict_online(self, h_freq_csi_history):
        h = np.asarray(h_freq_csi_history)
        if h.ndim != 8:
            raise ValueError("Configured WESN expects 8D CSI history tensor.")

        # align to [T, batch, rx_node, rx_ant, tx_node, tx_ant, freq, ofdm]
        h = h.transpose([0, 1, 2, 3, 4, 5, 7, 6])
        num_freq = h.shape[6]
        num_ofdm = h.shape[7]

        if h.shape[0] < 2:
            pred = np.zeros(h[0].shape, dtype=self.dtype)
            return pred.transpose([0, 1, 2, 3, 4, 6, 5])

        obs_seq = h[:, 0, 0, :, 0, :, :, :]
        T = obs_seq.shape[0]
        B = num_freq * num_ofdm

        obs_batch = np.transpose(obs_seq, (0, 3, 4, 1, 2)).reshape(T, B, self.N_r, self.N_t)

        # Mimic the test script behavior: solve LS readout online from the current
        # history window and use the last feature vector for one-step prediction.
        u_hist = self._window_sequence(obs_batch)
        s_hist = self._state_rollout(u_hist)

        self._fit_readout(states=s_hist[:-1], inputs=u_hist[:-1], targets=obs_batch[1:])

        if self.enable_skip_connections:
            feat_last = np.concatenate([s_hist[-1], u_hist[-1]], axis=-1)
        else:
            feat_last = s_hist[-1]
        y_last = feat_last @ self.W_out.T

        pred = np.zeros(h[0].shape, dtype=self.dtype)

        last = y_last.reshape(num_freq, num_ofdm, self.N_r, self.N_t)
        last = np.transpose(last, (2, 3, 0, 1))
        pred[0, 0, :, 0, :, :, :] = last

        return pred.transpose([0, 1, 2, 3, 4, 6, 5])

    def predict(self, h_freq_csi_history, err_var_history=None):
        if not self.is_frozen:
            self.fit_offline(h_freq_csi_history, err_var_history)
        return self.predict_online(h_freq_csi_history)
    
def _freeze_predictor_weights(predictor):
    predictor.enable_kalman_weight_config = False
    predictor.is_frozen = True


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
            predictor.fit_offline(curr_h, curr_e)
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