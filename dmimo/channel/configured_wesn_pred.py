import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.linalg import block_diag, solve_discrete_lyapunov
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import lsqr
from scipy.sparse import coo_matrix

from dmimo.channel.kalman_filter_pred import kalman_filter_pred
from dmimo.channel.complexity_instrumentation import measure_phase


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
    max_mag = float(np.max(np.abs(out))) if out.size > 0 else 0.0
    if np.isfinite(max_mag) and max_mag > float(max_radius) and max_mag > 1e-12:
        out *= float(max_radius) / max_mag
    return out


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


def _psd_factor(matrix, relative_tol=1e-12):
    """Return a square-root factor of a Hermitian PSD matrix."""
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    largest = max(float(np.max(values)), 0.0)
    if largest == 0.0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)
    keep = values > float(relative_tol) * largest
    return vectors[:, keep] * np.sqrt(values[keep])[None, :]


def square_root_balanced_truncate(
    a,
    b,
    c,
    d,
    order=None,
    energy_threshold=None,
    rank_tol=1e-12,
):
    """Return an order-reduced stable discrete-time realization."""
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a))))
    if spectral_radius >= 1.0 - 1e-10:
        raise ValueError(
            "Balanced truncation requires a stable lifted system; "
            f"spectral radius is {spectral_radius:.9f}."
        )

    controllability = solve_discrete_lyapunov(a, b @ b.conj().T)
    observability = solve_discrete_lyapunov(a.conj().T, c.conj().T @ c)
    rc = _psd_factor(controllability, rank_tol)
    ro = _psd_factor(observability, rank_tol)
    if rc.shape[1] == 0 or ro.shape[1] == 0:
        raise ValueError("Lifted PCA mode has no controllable-observable dynamics.")

    u, hsv, vh = np.linalg.svd(ro.conj().T @ rc, full_matrices=False)
    numerical_rank = int(np.count_nonzero(hsv > float(rank_tol) * hsv[0]))
    if energy_threshold is not None:
        if not (0.0 < float(energy_threshold) <= 1.0):
            raise ValueError("Hankel-energy threshold must lie in (0, 1].")
        energy = hsv[:numerical_rank] ** 2
        cumulative = np.cumsum(energy) / max(float(np.sum(energy)), 1e-30)
        order = int(
            np.searchsorted(cumulative, float(energy_threshold), side="left")
        ) + 1
    if order is None:
        raise ValueError("Specify either a balanced order or an energy threshold.")
    order = int(order)
    if order > numerical_rank:
        raise ValueError(
            f"Requested balanced order {order}, but the lifted mode has only "
            f"{numerical_rank} nonzero Hankel singular values."
        )

    inv_sqrt = 1.0 / np.sqrt(hsv[:order])
    right_projection = (rc @ vh.conj().T[:, :order]) * inv_sqrt[None, :]
    left_projection = (ro @ u[:, :order]) * inv_sqrt[None, :]
    return (
        left_projection.conj().T @ a @ right_projection,
        left_projection.conj().T @ b,
        c @ right_projection,
        d.copy(),
        hsv[:numerical_rank],
        2.0 * float(np.sum(hsv[order:numerical_rank])),
    )

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
        self.online_update = str(getattr(rc_config, "wesn_online_update", "batch_ridge")).lower()
        self.enable_residue_low_rank = bool(
            getattr(rc_config, "enable_residue_low_rank", False)
        )
        self.enable_balanced_truncation = bool(
            getattr(rc_config, "enable_balanced_truncation", False)
        )
        self.enable_balanced_hankel_truncation = bool(
            getattr(rc_config, "enable_balanced_hankel_truncation", False)
        )
        self.balanced_hankel_energy_threshold = float(
            getattr(rc_config, "balanced_hankel_energy_threshold", 0.90)
        )
        if self.enable_balanced_hankel_truncation:
            self.enable_balanced_truncation = True
        if self.enable_residue_low_rank and self.enable_balanced_truncation:
            raise ValueError(
                "Residue low-rank truncation and balanced truncation are "
                "mutually exclusive WESN configurations."
            )
        if not (0.0 < self.balanced_hankel_energy_threshold <= 1.0):
            raise ValueError(
                "balanced_hankel_energy_threshold must lie in (0, 1]."
            )
        self.residue_energy_threshold = float(
            getattr(rc_config, "residue_energy_threshold", 0.95)
        )
        self.reservoir_readout_reg = float(
            getattr(rc_config, "reservoir_readout_regularization", 1e-2)
        )
        self.skip_readout_reg = float(
            getattr(rc_config, "skip_readout_regularization", self.esn_ls_reg)
        )
        self.low_rank_readout_mode = str(
            getattr(rc_config, "wesn_lite_readout_mode", "centered_ridge")
        ).lower().replace("-", "_")
        self.wesn_lite_subcarriers_per_rb = int(
            getattr(rc_config, "wesn_lite_subcarriers_per_rb", 12)
        )
        if self.wesn_lite_subcarriers_per_rb < 1:
            raise ValueError("wesn_lite_subcarriers_per_rb must be positive.")
        if not (0.0 < self.residue_energy_threshold <= 1.0):
            raise ValueError(
                "residue_energy_threshold must be in (0, 1], got "
                f"{self.residue_energy_threshold}."
            )
        if self.reservoir_readout_reg < 0.0 or self.skip_readout_reg < 0.0:
            raise ValueError("Readout regularization values must be nonnegative.")
        if self.low_rank_readout_mode not in ("matched_ridge", "centered_ridge"):
            raise ValueError(
                "wesn_lite_readout_mode must be 'matched_ridge' or "
                f"'centered_ridge', got {self.low_rank_readout_mode!r}."
            )
        self.residue_ranks = []
        self.residue_retained_energy = []
        self.W_out_reference = None
        self.predictor_complexity_metrics = {"schema_version": 1, "phases": {}}

        self.input_dim = int(self.N_r * (self.N_t * self.window_length if self.enable_window else self.N_t))
        if self.enable_residue_low_rank or self.enable_balanced_hankel_truncation:
            # This is only a provisional size. Configuration replaces it with
            # sum_{m,k} r_{m,k}, which may legitimately be smaller than PD.
            self.state_dim = int(max(1, self.esn_m * self.esn_k))
        else:
            self.state_dim = int(max(self.input_dim, self.esn_m * self.esn_k * self.N_r * self.N_t))
        self.output_dim = int(self.N_r * self.N_t)
        self.feature_dim = int(self.state_dim + self.input_dim if self.enable_skip_connections else self.state_dim)
        self.predictor_complexity_metrics.update(
            {
                "method": (
                    "wesn_lite"
                    if self.enable_residue_low_rank
                    else "configured_wesn_balanced_lite"
                    if self.enable_balanced_hankel_truncation
                    else "configured_wesn_balanced"
                    if self.enable_balanced_truncation
                    else "configured_wesn"
                ),
                "input_dimension": self.input_dim,
                "state_dimension": self.state_dim,
                "output_dimension": self.output_dim,
                "feature_dimension": self.feature_dim,
                "online_update": self.online_update,
                "configured_modes": self.esn_m,
                "configured_poles_per_mode": self.esn_k,
                "residue_low_rank": self.enable_residue_low_rank,
                "balanced_truncation": self.enable_balanced_truncation,
                "balanced_hankel_truncation": (
                    self.enable_balanced_hankel_truncation
                ),
                "balanced_hankel_energy_threshold": (
                    self.balanced_hankel_energy_threshold
                ),
                "residue_energy_threshold": self.residue_energy_threshold,
                "reservoir_readout_regularization": self.reservoir_readout_reg,
                "skip_readout_regularization": self.skip_readout_reg,
                "readout_objective": (
                    self.low_rank_readout_mode
                    if self.enable_residue_low_rank
                    else "matched_ridge"
                ),
                "resource_block_averaging": True,
                "persistent_online_state": True,
                "subcarriers_per_resource_block": (
                    self.wesn_lite_subcarriers_per_rb
                ),
            }
        )

        self.RS = np.random.RandomState(10)
        self.is_frozen = False
        self.init_weights()
        self.reset_online_state()

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

    def _average_resource_grid_for_wesn_lite(self, values):
        """Average OFDM symbols and contiguous subcarriers into RB samples."""
        values = np.asarray(values)
        if values.ndim < 2:
            raise ValueError("Resource-grid input must have symbol and subcarrier axes.")

        symbol_average = np.mean(values, axis=-2, keepdims=True)
        num_subcarriers = int(symbol_average.shape[-1])
        rb_width = self.wesn_lite_subcarriers_per_rb
        resource_blocks = [
            np.mean(symbol_average[..., start : start + rb_width], axis=-1, keepdims=True)
            for start in range(0, num_subcarriers, rb_width)
        ]
        if not resource_blocks:
            raise ValueError("WESN-Lite requires at least one subcarrier.")
        return np.concatenate(resource_blocks, axis=-1).astype(
            values.dtype, copy=False
        )

    def _expand_wesn_lite_resource_grid(
        self,
        values,
        num_ofdm_symbols,
        num_subcarriers,
    ):
        """Broadcast RB predictions back to the simulator's OFDM grid."""
        expanded = np.repeat(values, int(num_ofdm_symbols), axis=-2)
        expanded = np.repeat(
            expanded,
            self.wesn_lite_subcarriers_per_rb,
            axis=-1,
        )
        return expanded[..., : int(num_subcarriers)].astype(
            values.dtype, copy=False
        )

    def _activation(self, x):
        return apply_complex_activation(x, self.esn_activation)

    def reset_online_state(self):
        """Discard the continuous WESN state and rolling feature cache."""
        self._online_state = None
        self._online_states = None
        self._online_inputs = None
        self._online_observations = None
        self.num_reservoir_updates_last_predict = 0

    def _state_step(self, state, u_t):
        # The low-rank input factor is already Sigma*V^H. Applying the generic
        # random-ESN input scaling again would alter that configured factor.
        projected_input = (
            u_t
            if getattr(self, "enable_residue_low_rank", False)
            else self.input_scale * u_t
        )
        if self.W_res.ndim == 1:
            recurrent = state * self.W_res[None, :]
        else:
            recurrent = state @ self.W_res.T
        return self._activation((projected_input @ self.W_in.T) + recurrent).astype(
            self.dtype, copy=False
        )

    def _state_rollout(self, u_seq):
        # u_seq: [T, B, input_dim]
        T, B, _ = u_seq.shape
        state = np.zeros((B, self.state_dim), dtype=self.dtype)
        states = np.zeros((T, B, self.state_dim), dtype=self.dtype)
        for t in range(T):
            state = self._state_step(state, u_seq[t])
            states[t] = state
        return states.astype(self.dtype)

    def _can_advance_online_cache(self, obs_batch):
        if (
            self._online_state is None
            or self._online_states is None
            or self._online_inputs is None
            or self._online_observations is None
        ):
            return False
        if self._online_observations.shape != obs_batch.shape:
            return False
        if self._online_state.shape != (obs_batch.shape[1], self.state_dim):
            return False
        # Consecutive simulation calls slide the T-sample history by one. Exact
        # equality is appropriate because overlapping estimates are loaded from
        # the same stored tensors; a discontinuity intentionally resets state.
        return np.array_equal(
            obs_batch[:-1],
            self._online_observations[1:],
        )

    def _persistent_state_features(self, obs_batch, u_hist):
        """Return rolling WESN states, advancing once when histories overlap."""
        if self._can_advance_online_cache(obs_batch):
            state_new = self._state_step(self._online_state, u_hist[-1])
            states = np.concatenate(
                [self._online_states[1:], state_new[None, ...]], axis=0
            )
            inputs = np.concatenate(
                [self._online_inputs[1:], u_hist[-1:]], axis=0
            )
            self.num_reservoir_updates_last_predict = 1
        else:
            states = self._state_rollout(u_hist)
            inputs = u_hist
            state_new = states[-1]
            self.num_reservoir_updates_last_predict = int(u_hist.shape[0])

        self._online_state = np.asarray(state_new, dtype=self.dtype).copy()
        self._online_states = np.asarray(states, dtype=self.dtype).copy()
        self._online_inputs = np.asarray(inputs, dtype=self.dtype).copy()
        self._online_observations = np.asarray(obs_batch, dtype=self.dtype).copy()
        self.predictor_complexity_metrics.update(
            {
                "num_reservoir_updates_last_predict": int(
                    self.num_reservoir_updates_last_predict
                ),
                "online_feature_cache_length": int(states.shape[0]),
            }
        )
        return states, inputs

    def _fit_readout(self, states, inputs, targets):
        if self.enable_skip_connections:
            feat = np.concatenate([states, inputs], axis=-1).reshape(-1, self.feature_dim)
        else:
            feat = states.reshape(-1, self.feature_dim)
        y = targets.reshape(-1, self.output_dim)

        # Forming Z^H Z squares the condition number of Z.  In the production
        # path the reservoir tensors are complex64, and the resulting ridge
        # systems can have condition numbers of 1e6 or larger.  Solving those
        # normal equations in complex64 can therefore produce sporadic,
        # extremely large readouts even though the ridge problem itself is
        # well-defined.  Accumulate and solve in complex128, then retain the
        # configured inference dtype for W_out.
        solve_dtype = np.complex128
        feat_solve = np.asarray(feat, dtype=solve_dtype)
        y_solve = np.asarray(y, dtype=solve_dtype)
        if (
            self.enable_residue_low_rank
            and self.low_rank_readout_mode == "centered_ridge"
            and self.W_out_reference is not None
        ):
            # Centered ridge preserves the configured truncated transfer while
            # allowing a data-driven residual correction.  The normalized-loss
            # convention produces N*lambda in the unnormalized normal equations.
            num_samples = int(feat.shape[0])
            regularization = np.full(
                (self.feature_dim,),
                num_samples * self.reservoir_readout_reg,
                dtype=np.float64,
            )
            if self.enable_skip_connections:
                regularization[self.state_dim :] = (
                    num_samples * self.skip_readout_reg
                )
            gram = feat_solve.conj().T @ feat_solve + np.diag(regularization)
            reference_t = np.asarray(
                self.W_out_reference.T, dtype=solve_dtype
            )
            rhs = (
                feat_solve.conj().T @ y_solve
                + regularization[:, None] * reference_t
            )
            try:
                fitted_t = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                fitted_t = np.linalg.pinv(gram) @ rhs
            self.W_out = fitted_t.T.astype(self.dtype, copy=False)
            residual = self.W_out - self.W_out_reference
            self.predictor_complexity_metrics.update(
                {
                    "readout_reference_fro_norm": float(
                        np.linalg.norm(self.W_out_reference, ord="fro")
                    ),
                    "readout_residual_fro_norm": float(
                        np.linalg.norm(residual, ord="fro")
                    ),
                    "readout_total_fro_norm": float(
                        np.linalg.norm(self.W_out, ord="fro")
                    ),
                }
            )
            return

        gram = (
            feat_solve.conj().T @ feat_solve
            + self.esn_ls_reg * np.eye(self.feature_dim, dtype=solve_dtype)
        )
        rhs = feat_solve.conj().T @ y_solve
        if (
            getattr(self, "esn_diagnostics", False)
            and "readout_gram_condition_first"
            not in self.predictor_complexity_metrics
        ):
            self.predictor_complexity_metrics["readout_gram_condition_first"] = float(
                np.linalg.cond(gram)
            )
        try:
            fitted_t = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            fitted_t = np.linalg.pinv(gram) @ rhs
        self.W_out = fitted_t.T.astype(self.dtype, copy=False)
        self.predictor_complexity_metrics.update(
            {
                "readout_solve_dtype": "complex128",
                "readout_total_fro_norm": float(
                    np.linalg.norm(fitted_t.T, ord="fro")
                ),
                "readout_max_abs": float(np.max(np.abs(fitted_t))),
            }
        )

    @staticmethod
    def _rank_for_energy(singular_values, threshold):
        """Smallest SVD rank retaining ``threshold`` Frobenius energy."""
        singular_values = np.asarray(singular_values, dtype=np.float64)
        energy = singular_values**2
        total = float(np.sum(energy))
        if singular_values.size == 0:
            return 0, 1.0
        if not np.isfinite(total) or total <= np.finfo(np.float64).eps:
            return 1, 1.0
        cumulative = np.cumsum(energy) / total
        rank = int(np.searchsorted(cumulative, threshold, side="left")) + 1
        rank = min(max(1, rank), singular_values.size)
        return rank, float(cumulative[rank - 1])

    def _configure_low_rank_reservoir(self, poles_all, residues_all, d):
        """Realize each D-by-D pole residue with adaptive SVD factors."""
        recurrent_weights = []
        input_rows = []
        reference_output_columns = []
        residue_ranks = []
        retained_energy = []

        for mode_idx in range(poles_all.shape[0]):
            for pole_idx in range(poles_all.shape[1]):
                # vectorize_transfer_samples() stores vec(C) as C.T.reshape(-1).
                residue = residues_all[mode_idx, :, pole_idx].reshape(d, d).T
                u, singular_values, vh = np.linalg.svd(
                    residue, full_matrices=False
                )
                rank, energy = self._rank_for_energy(
                    singular_values, self.residue_energy_threshold
                )
                residue_ranks.append(rank)
                retained_energy.append(energy)

                for factor_idx in range(rank):
                    recurrent_weights.append(poles_all[mode_idx, pole_idx])
                    input_row = np.zeros((self.input_dim,), dtype=self.dtype)
                    # vh[a] is v_a^H. Embed it in the current-observation
                    # (last) D-dimensional block of the P-sample input window.
                    input_row[-d:] = (
                        singular_values[factor_idx] * vh[factor_idx]
                    ).astype(self.dtype, copy=False)
                    input_rows.append(input_row)
                    reference_output_columns.append(
                        u[:, factor_idx].astype(self.dtype, copy=False)
                    )

        if not recurrent_weights:
            raise RuntimeError("Low-rank residue configuration produced no states.")

        self.W_res = np.asarray(recurrent_weights, dtype=self.dtype)
        self.W_in = np.asarray(input_rows, dtype=self.dtype)
        self.state_dim = int(self.W_res.size)
        self.feature_dim = int(
            self.state_dim + self.input_dim
            if self.enable_skip_connections
            else self.state_dim
        )
        self.W_out_reference = np.zeros(
            (self.output_dim, self.feature_dim), dtype=self.dtype
        )
        self.W_out_reference[:, : self.state_dim] = np.stack(
            reference_output_columns, axis=1
        )
        self.W_out = self.W_out_reference.copy()
        self.residue_ranks = [int(rank) for rank in residue_ranks]
        self.residue_retained_energy = [float(value) for value in retained_energy]

        unique, counts = np.unique(self.residue_ranks, return_counts=True)
        rank_mode = int(unique[np.argmax(counts)])
        rank_histogram = {
            str(int(rank)): int(count)
            for rank, count in zip(unique, counts)
        }
        self.predictor_complexity_metrics.update(
            {
                "state_dimension": self.state_dim,
                "feature_dimension": self.feature_dim,
                "residue_ranks": self.residue_ranks,
                "residue_rank_mean": float(np.mean(self.residue_ranks)),
                "residue_rank_median": float(np.median(self.residue_ranks)),
                "residue_rank_mode": rank_mode,
                "residue_rank_min": int(np.min(self.residue_ranks)),
                "residue_rank_max": int(np.max(self.residue_ranks)),
                "residue_rank_histogram": rank_histogram,
                "residue_retained_energy_mean": float(
                    np.mean(self.residue_retained_energy)
                ),
                "residue_retained_energy_min": float(
                    np.min(self.residue_retained_energy)
                ),
                "readout_reference_fro_norm": float(
                    np.linalg.norm(self.W_out_reference, ord="fro")
                ),
            }
        )

    def _configure_balanced_reservoir(self, realizations, alpha, d):
        """Balance and truncate each exact lifted empirical PCA component."""
        reduced_order = int(self.esn_k * d)
        recurrent_weights = []
        input_rows = []
        reference_output_columns = []
        hankel_values = []
        error_bounds = []
        raw_orders = []
        selected_orders = []
        retained_energies = []

        for mode_idx in range(self.esn_m):
            coefficients = alpha[:, mode_idx]
            # PCA was applied to centered transfer samples. Subtracting the
            # coefficient mean gives the exact continuous rational lift of the
            # corresponding centered PCA mode.
            weights = coefficients - np.mean(coefficients)
            a_lift = block_diag(
                *(system[0] for system in realizations)
            ).astype(np.complex128)
            b_lift = np.vstack([system[1] for system in realizations])
            c_lift = np.hstack(
                [weight * system[2] for weight, system in zip(weights, realizations)]
            )
            d_lift = sum(
                (weight * system[3] for weight, system in zip(weights, realizations)),
                start=np.zeros_like(realizations[0][3]),
            )
            raw_orders.append(int(a_lift.shape[0]))

            a_red, b_red, c_red, _, hsv, error_bound = (
                square_root_balanced_truncate(
                    a_lift,
                    b_lift,
                    c_lift,
                    d_lift,
                    order=(
                        None
                        if self.enable_balanced_hankel_truncation
                        else reduced_order
                    ),
                    energy_threshold=(
                        self.balanced_hankel_energy_threshold
                        if self.enable_balanced_hankel_truncation
                        else None
                    ),
                )
            )
            selected_order = int(a_red.shape[0])
            hankel_energy = np.asarray(hsv, dtype=np.float64) ** 2
            retained_energy = float(
                np.sum(hankel_energy[:selected_order])
                / max(float(np.sum(hankel_energy)), 1e-30)
            )
            selected_orders.append(selected_order)
            retained_energies.append(retained_energy)
            poles, right_vectors = np.linalg.eig(a_red)
            inverse_right = np.linalg.inv(right_vectors)
            if np.min(np.abs(poles)) < 1e-10:
                raise RuntimeError(
                    "A balanced WESN pole is numerically zero; the current "
                    "simple-IIR reservoir requires nonzero poles."
                )

            for pole_idx, pole in enumerate(poles):
                conventional_residue = np.outer(
                    c_red @ right_vectors[:, pole_idx],
                    inverse_right[pole_idx] @ b_red,
                )
                residue = conventional_residue / pole
                u, singular_values, vh = np.linalg.svd(
                    residue, full_matrices=False
                )
                leading = float(singular_values[0])
                tail = float(np.linalg.norm(singular_values[1:]))
                if tail > 1e-6 * max(leading, 1e-15):
                    raise RuntimeError(
                        "A balanced WESN pole residue is not numerically rank "
                        f"one (relative tail {tail / max(leading, 1e-15):.3e})."
                    )
                input_row = np.zeros((self.input_dim,), dtype=self.dtype)
                input_row[-d:] = (leading * vh[0]).astype(
                    self.dtype, copy=False
                )
                recurrent_weights.append(pole)
                input_rows.append(input_row)
                reference_output_columns.append(
                    u[:, 0].astype(self.dtype, copy=False)
                )

            hankel_values.append(hsv)
            error_bounds.append(error_bound)

        self.W_res = np.asarray(recurrent_weights, dtype=self.dtype)
        self.W_in = np.asarray(input_rows, dtype=self.dtype)
        self.state_dim = int(self.W_res.size)
        self.feature_dim = int(
            self.state_dim + self.input_dim
            if self.enable_skip_connections
            else self.state_dim
        )
        self.W_out_reference = np.zeros(
            (self.output_dim, self.feature_dim), dtype=self.dtype
        )
        self.W_out_reference[:, : self.state_dim] = np.stack(
            reference_output_columns, axis=1
        )
        self.W_out = self.W_out_reference.copy()
        self.residue_ranks = [1] * self.state_dim
        self.residue_retained_energy = [1.0] * self.state_dim
        self.predictor_complexity_metrics.update(
            {
                "state_dimension": self.state_dim,
                "feature_dimension": self.feature_dim,
                "balanced_order_per_mode": (
                    selected_orders
                    if self.enable_balanced_hankel_truncation
                    else reduced_order
                ),
                "balanced_orders_per_mode": selected_orders,
                "balanced_retained_hankel_energy_per_mode": retained_energies,
                "balanced_raw_orders": raw_orders,
                "balanced_hankel_singular_values": [
                    np.asarray(values, dtype=float).tolist()
                    for values in hankel_values
                ],
                "balanced_error_bounds": [float(value) for value in error_bounds],
                "residue_ranks": self.residue_ranks,
                "residue_rank_mean": 1.0,
                "residue_rank_min": 1,
                "residue_rank_max": 1,
                "readout_reference_fro_norm": float(
                    np.linalg.norm(self.W_out_reference, ord="fro")
                ),
            }
        )


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
        realizations = []
        max_start = max(1, t_dec - history_len)
        for s in range(max_start):
            y_hist_chunk = u_seq[s: s + history_len].astype(np.complex64)
            with measure_phase(self, "configuration_ar"):
                a_blocks, q_proc = kf_helper._estimate_ar_p_q_joint(y_hist_chunk, p_eff)
            a_blocks = [a.conj() for a in a_blocks]
            f_aug, q_aug = kf_helper._build_augmented_system(a_blocks, q_proc)
            h_mat = build_augmented_obs_matrix(d, p_eff)
            r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex64))
            with measure_phase(self, "configuration_riccati"):
                _, k_ss = solve_riccati_steady_state_complex(f_aug, q_aug, h_mat, r_mat)
            with measure_phase(self, "configuration_transfer_sampling"):
                h_samps = steady_state_predictor_transfer_samples_from_kalman(
                    f_aug,
                    k_ss,
                    d=d,
                    num_freqs=self.esn_num_freqs,
                )
            v_list.append(vectorize_transfer_samples(h_samps))
            if self.enable_balanced_truncation:
                a_p = f_aug - f_aug @ k_ss @ h_mat
                b_p = f_aug @ k_ss
                realizations.append(
                    (a_p, b_p, h_mat @ a_p, h_mat @ b_p)
                )

        if len(v_list) == 0:
            return

        v = np.stack(v_list, axis=1)
        _, vc, kvv = estimate_empirical_complex_covariance(v)

        m = self.esn_m

        if vc.shape[0] > vc.shape[1] and m <= vc.shape[1]:
            with measure_phase(self, "configuration_pca_svd"):
                u, svals, vh = np.linalg.svd(vc, full_matrices=False)
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

        if self.enable_balanced_truncation:
            alpha = vh.conj().T[:, idx] / svals[idx][None, :]
            with measure_phase(self, "configuration_balanced_truncation"):
                self._configure_balanced_reservoir(realizations, alpha, d)
            return

        value_dim = d * d
        poles_all = np.zeros((m, self.esn_k), dtype=np.complex64)
        residues_all = np.zeros((m, value_dim, self.esn_k), dtype=np.complex64)

        for j in range(m):
            q_col = q_eig[:, j]
            with measure_phase(self, "configuration_vector_fitting"):
                fit = fit_shared_denominator_vector_rational(
                    q_col=q_col,
                    num_freqs=self.esn_num_freqs,
                    value_dim=value_dim,
                    degree=self.esn_k,
                    omegas=omegas,
                )
            with measure_phase(self, "configuration_pole_residue_processing"):
                poles, residues = decompose_rp_fit_into_first_order(
                    fit=fit,
                    q_col=q_col,
                    num_freqs=self.esn_num_freqs,
                    value_dim=value_dim,
                    omegas=omegas,
                )
            poles_all[j] = poles
            residues_all[j] = residues

        with measure_phase(self, "configuration_pole_stabilization"):
            pole_max_mag = float(np.max(np.abs(poles_all))) if poles_all.size > 0 else 0.0
            # Use the same configured pole bank for the full and low-rank
            # reservoirs so the lite comparison isolates residue truncation.
            if (
                np.isfinite(pole_max_mag)
                and pole_max_mag > float(self.spectral_radius)
                and pole_max_mag > 1e-12
            ):
                poles_all = poles_all * (float(self.spectral_radius) / pole_max_mag)

        with measure_phase(self, "configuration_residue_mapping"):
            if self.enable_residue_low_rank:
                self._configure_low_rank_reservoir(poles_all, residues_all, d)
            else:
                w_res = np.tile(poles_all.reshape(-1), int(np.ceil(self.state_dim / (m * self.esn_k))))[: self.state_dim]
                # The configured recurrence is diagonal. Store only its diagonal so
                # inference and memory scale as O(R), matching the analytical model.
                self.W_res = w_res.astype(self.dtype)

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

        # Every configured-WESN variant operates on one value per resource
        # block. Predictions are expanded back to the simulator's original
        # OFDM-symbol/subcarrier grid in predict_online().
        original_num_symbols = int(h.shape[-2])
        original_num_subcarriers = int(h.shape[-1])
        h = self._average_resource_grid_for_wesn_lite(h)
        if e is not None:
            e = self._average_resource_grid_for_wesn_lite(e)
        self.predictor_complexity_metrics.update(
            {
                "original_num_ofdm_symbols": original_num_symbols,
                "original_num_subcarriers": original_num_subcarriers,
                "model_num_ofdm_symbols": int(h.shape[-2]),
                "model_num_resource_blocks": int(h.shape[-1]),
            }
        )

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
                if self.enable_residue_low_rank or self.enable_balanced_truncation:
                    err_seq = e_batch.reshape(T, B, self.output_dim)
                else:
                    err_seq = self._window_sequence(e_batch)
            # The low-rank realization factorizes the D-by-D residues defined
            # in the configured-WESN derivation. The resulting D-dimensional
            # input factors are then embedded into the current block of the
            # PD-dimensional WESN window. Keep the legacy configured-WESN path
            # unchanged for comparison with its existing results.
            configuration_seq = (
                y_in_batch.reshape(T, B, self.output_dim)
                if self.enable_residue_low_rank or self.enable_balanced_truncation
                else u_seq
            )
            self._configure_weights_from_kalman_stats(
                configuration_seq, err_var_seq=err_seq
            )

        self.is_frozen = True
        self.reset_online_state()

    def predict_online(self, h_freq_csi_history):
        h = np.asarray(h_freq_csi_history)
        if h.ndim != 8:
            raise ValueError("Configured WESN expects 8D CSI history tensor.")

        original_num_symbols = int(h.shape[-2])
        original_num_subcarriers = int(h.shape[-1])
        h = self._average_resource_grid_for_wesn_lite(h)

        # align to [T, batch, rx_node, rx_ant, tx_node, tx_ant, freq, ofdm]
        h = h.transpose([0, 1, 2, 3, 4, 5, 7, 6])
        num_freq = h.shape[6]
        num_ofdm = h.shape[7]

        if h.shape[0] < 2:
            self.reset_online_state()
            pred = np.zeros(h[0].shape, dtype=self.dtype)
            result = pred.transpose([0, 1, 2, 3, 4, 6, 5])
            result = self._expand_wesn_lite_resource_grid(
                result,
                original_num_symbols,
                original_num_subcarriers,
            )
            return result

        obs_seq = h[:, 0, 0, :, 0, :, :, :]
        T = obs_seq.shape[0]
        B = num_freq * num_ofdm

        obs_batch = np.transpose(obs_seq, (0, 3, 4, 1, 2)).reshape(T, B, self.N_r, self.N_t)

        # Both configured WESN realizations retain a continuous reservoir
        # state and the last T feature blocks. A sliding history advances the
        # recurrence once; a discontinuity falls back to a zero-state replay.
        with measure_phase(self, "inference_reservoir_rollout"):
            u_hist = self._window_sequence(obs_batch)
            s_hist, u_hist = self._persistent_state_features(
                obs_batch, u_hist
            )

        with measure_phase(self, "online_update_batch_ridge"):
            self._fit_readout(states=s_hist[:-1], inputs=u_hist[:-1], targets=obs_batch[1:])

        if self.enable_skip_connections:
            feat_last = np.concatenate([s_hist[-1], u_hist[-1]], axis=-1)
        else:
            feat_last = s_hist[-1]
        with measure_phase(self, "inference_readout"):
            y_last = feat_last @ self.W_out.T

        pred = np.zeros(h[0].shape, dtype=self.dtype)

        last = y_last.reshape(num_freq, num_ofdm, self.N_r, self.N_t)
        last = np.transpose(last, (2, 3, 0, 1))
        pred[0, 0, :, 0, :, :, :] = last

        result = pred.transpose([0, 1, 2, 3, 4, 6, 5])
        result = self._expand_wesn_lite_resource_grid(
            result,
            original_num_symbols,
            original_num_subcarriers,
        )
        return result

    def predict(self, h_freq_csi_history, err_var_history=None):
        if not self.is_frozen:
            self.fit_offline(h_freq_csi_history, err_var_history)
        return self.predict_online(h_freq_csi_history)
    
def _freeze_predictor_weights(predictor):
    predictor.enable_kalman_weight_config = False
    predictor.is_frozen = True


def _configured_link_antennas(
    tx_node_idx,
    rx_node_idx,
    num_bs_ant,
    num_ue_ant,
):
    tx_ant_idx = np.arange(0, num_bs_ant) if tx_node_idx == 0 else np.arange(
        num_bs_ant + (tx_node_idx - 1) * num_ue_ant,
        num_bs_ant + tx_node_idx * num_ue_ant,
    )
    rx_ant_idx = np.arange(0, num_bs_ant) if rx_node_idx == 0 else np.arange(
        num_bs_ant + (rx_node_idx - 1) * num_ue_ant,
        num_bs_ant + rx_node_idx * num_ue_ant,
    )
    return rx_ant_idx, tx_ant_idx


def _predict_configured_link_worker(args):
    (
        tx_node_idx,
        rx_node_idx,
        predictor,
        h_freq_csi_history,
        err_var_csi_history,
        num_bs_ant,
        num_ue_ant,
    ) = args
    rx_ant_idx, tx_ant_idx = _configured_link_antennas(
        tx_node_idx,
        rx_node_idx,
        num_bs_ant,
        num_ue_ant,
    )

    curr_h = h_freq_csi_history[:, :, :, rx_ant_idx, :, ...]
    curr_h = curr_h[:, :, :, :, :, tx_ant_idx, ...]

    curr_e = None
    if err_var_csi_history is not None:
        curr_e = err_var_csi_history[:, :, :, rx_ant_idx, :, ...]
        curr_e = curr_e[:, :, :, :, :, tx_ant_idx, ...]

    tmp = np.asarray(predictor.predict(curr_h, curr_e))
    return (
        (tx_node_idx, rx_node_idx),
        rx_ant_idx,
        tx_ant_idx,
        tmp,
        predictor.W_out,
        predictor.predictor_complexity_metrics,
    )


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
            rx_ant_idx, tx_ant_idx = _configured_link_antennas(
                tx_node_idx,
                rx_node_idx,
                num_bs_ant,
                num_ue_ant,
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

    links = [
        (tx_node_idx, rx_node_idx)
        for tx_node_idx in range(ns3cfg.num_txue_sel + 1)
        for rx_node_idx in range(ns3cfg.num_rxue_sel + 1)
    ]
    for tx_node_idx, rx_node_idx in links:
        result = _predict_configured_link_worker(
            (
                tx_node_idx,
                rx_node_idx,
                configured_predictors[(tx_node_idx, rx_node_idx)],
                h_freq_csi_history,
                err_var_csi_history,
                num_bs_ant,
                num_ue_ant,
            )
        )
        _, rx_ant_idx, tx_ant_idx, tmp, _, _ = result
        rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
        h_freq_csi[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

    return h_freq_csi


def predict_all_links_with_configured(
    h_freq_csi_history,
    configured_predictors,
    ns3cfg,
    num_bs_ant=4,
    num_ue_ant=2,
    err_var_csi_history=None,
    max_workers=4,
    executor=None,
):
    """Predict configured-WESN/WESN-Lite links in shared-memory threads.

    Pass a persistent ``ThreadPoolExecutor`` through ``executor`` to reuse
    already-started workers across prediction slots. When omitted, a temporary
    pool is created and shut down for this call.
    """
    base_history = np.asarray(h_freq_csi_history)
    err_var_history = (
        None if err_var_csi_history is None else np.asarray(err_var_csi_history)
    )
    h_freq_csi = np.zeros(base_history[0, ...].shape, dtype=base_history.dtype)

    links = [
        (tx_node_idx, rx_node_idx)
        for tx_node_idx in range(ns3cfg.num_txue_sel + 1)
        for rx_node_idx in range(ns3cfg.num_rxue_sel + 1)
    ]
    if max_workers is None:
        max_workers = min(len(links), os.cpu_count() or 1)
    max_workers = max(1, min(int(max_workers), len(links)))

    if max_workers == 1:
        return predict_all_links_with_configured_simple(
            base_history,
            configured_predictors,
            ns3cfg,
            num_bs_ant=num_bs_ant,
            num_ue_ant=num_ue_ant,
            err_var_csi_history=err_var_history,
        )

    tasks = [
        (
            tx_node_idx,
            rx_node_idx,
            configured_predictors[(tx_node_idx, rx_node_idx)],
            base_history,
            err_var_history,
            num_bs_ant,
            num_ue_ant,
        )
        for tx_node_idx, rx_node_idx in links
    ]
    def collect_results(active_executor):
        futures = [
            active_executor.submit(_predict_configured_link_worker, task)
            for task in tasks
        ]
        for future in as_completed(futures):
            (
                link,
                rx_ant_idx,
                tx_ant_idx,
                tmp,
                updated_w_out,
                updated_metrics,
            ) = future.result()
            # Prediction performs an online readout update. Carry that mutable
            # state back from the child so the next slot starts where this one
            # finished and per-link instrumentation is not lost.
            predictor = configured_predictors[link]
            predictor.W_out = updated_w_out
            predictor.predictor_complexity_metrics = updated_metrics

            rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
            h_freq_csi[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(
                2, 4, 0, 1, 3, 5, 6
            )

    if executor is None:
        with ThreadPoolExecutor(max_workers=max_workers) as temporary_executor:
            collect_results(temporary_executor)
    else:
        collect_results(executor)

    return h_freq_csi
