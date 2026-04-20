import argparse
from pathlib import Path
import re
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)


from dmimo.channel.kalman_filter_pred import kalman_filter_pred


def discover_slot_files(folder: Path) -> list[tuple[int, Path]]:
    pat = re.compile(r"dmimochans_(\d+)\.npz$")
    out: list[tuple[int, Path]] = []
    for p in folder.glob("dmimochans_*.npz"):
        m = pat.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def load_clean_p2p_channels(
    ns3_folder: Path,
    drop_idx: int,
    mobility: str,
    start_slot: int,
    end_slot: int,
    feedback_delay: int,
    rx_ant: int,
    tx_ant: int,
) -> tuple[np.ndarray, np.ndarray]:
    folder = ns3_folder / f"channels_{mobility}_{drop_idx}"
    if not folder.exists():
        raise FileNotFoundError(f"Channel folder not found: {folder}")

    files = discover_slot_files(folder)
    if not files:
        raise FileNotFoundError(f"No channels_*.npz files found under: {folder}")

    by_slot = {slot: path for slot, path in files}
    selected_slots = [s for s in range(start_slot, end_slot + 1, feedback_delay) if s in by_slot]
    if len(selected_slots) < 3:
        raise RuntimeError(
            f"Too few decimated slots found. Requested {start_slot}:{end_slot}:{feedback_delay}, "
            f"found {len(selected_slots)} in {folder}."
        )

    h_list = []
    for s in selected_slots:
        with np.load(by_slot[s]) as data:
            hdm = data["Hdm"]  # [all_rx_ant, all_tx_ant, num_syms, num_sc]
        h_p2p = hdm[:rx_ant, :tx_ant, :, :].astype(np.complex128)
        h_list.append(h_p2p)

    # [T_decimated, rx_ant, tx_ant, num_syms, num_sc]
    h_seq = np.stack(h_list, axis=0)
    return h_seq, np.asarray(selected_slots, dtype=int)


def add_complex_awgn(h: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    # h shape: [T, rx, tx, syms, sc]
    signal_power = float(np.mean(np.abs(h) ** 2))
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = signal_power / max(snr_lin, 1e-15)
    noise = (
        rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape)
    ) * np.sqrt(noise_var / 2.0)
    return h + noise, noise_var


def channels_to_tiles(h_seq: np.ndarray) -> np.ndarray:
    # h_seq: [T, rx, tx, syms, sc]
    # tiles: [T, Ntiles, D] where Ntiles=syms*sc, D=rx*tx
    t_len, rx, tx, syms, sc = h_seq.shape
    return h_seq.transpose(0, 3, 4, 1, 2).reshape(t_len, syms * sc, rx * tx)


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
    pd = f_aug.shape[0]
    p_minus = np.eye(pd, dtype=np.complex128)

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


def full_kalman_predict_next(y_hist: np.ndarray, f_aug: np.ndarray, q_aug: np.ndarray, r_diag: np.ndarray) -> np.ndarray:
    t_len, d = y_hist.shape
    p = f_aug.shape[0] // d
    if t_len <= p:
        return y_hist[-1].astype(np.complex128)

    h_mat = build_augmented_obs_matrix(d, p)
    r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex128))

    # initialize state stack from earliest available history
    state_stack = [y_hist[p - 1 - k].astype(np.complex128) for k in range(p)]
    z_hat = np.concatenate(state_stack, axis=0)
    p_hat = np.eye(p * d, dtype=np.complex128) * (float(np.mean(r_diag)) + 1e-6)
    eye_pd = np.eye(p * d, dtype=np.complex128)

    for t in range(p, t_len):
        z_pred = f_aug @ z_hat
        p_pred = f_aug @ p_hat @ f_aug.conj().T + q_aug

        innov = y_hist[t].astype(np.complex128) - h_mat @ z_pred
        s = h_mat @ p_pred @ h_mat.conj().T + r_mat
        k = p_pred @ h_mat.conj().T @ np.linalg.pinv(s)

        z_hat = z_pred + k @ innov
        p_hat = (eye_pd - k @ h_mat) @ p_pred

    z_next = f_aug @ z_hat
    return z_next[:d]


def steady_kalman_predict_next(
    y_hist: np.ndarray,
    f_aug: np.ndarray,
    q_aug: np.ndarray,
    r_diag: np.ndarray,
) -> np.ndarray:
    t_len, d = y_hist.shape
    p = f_aug.shape[0] // d
    if t_len <= p:
        return y_hist[-1].astype(np.complex128)

    h_mat = build_augmented_obs_matrix(d, p)
    r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex128))
    _, k_ss = solve_riccati_steady_state_complex(f_aug, q_aug, h_mat, r_mat)

    state_stack = [y_hist[p - 1 - k].astype(np.complex128) for k in range(p)]
    z_hat = np.concatenate(state_stack, axis=0)

    for t in range(p, t_len):
        z_pred = f_aug @ z_hat
        innov = y_hist[t].astype(np.complex128) - h_mat @ z_pred
        z_hat = z_pred + k_ss @ innov

    z_next = f_aug @ z_hat
    return z_next[:d]

def _init_augmented_state_batch(y_hist_batch: np.ndarray, p: int) -> np.ndarray:
    # y_hist_batch: [T, Ntiles, D] -> state: [Ntiles, p*D]
    state_stack = y_hist_batch[p - 1 :: -1][:p].astype(np.complex128)
    ntiles, d = state_stack.shape[1], state_stack.shape[2]
    return state_stack.transpose(1, 0, 2).reshape(ntiles, p * d)


def full_kalman_predict_next_batch(
    y_hist_batch: np.ndarray,
    f_aug: np.ndarray,
    q_aug: np.ndarray,
    r_diag: np.ndarray,
) -> np.ndarray:
    t_len, ntiles, d = y_hist_batch.shape
    p = f_aug.shape[0] // d
    if t_len <= p:
        return y_hist_batch[-1].astype(np.complex128)

    pd = p * d
    r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex128))
    z_hat = _init_augmented_state_batch(y_hist_batch, p)
    p_hat = np.broadcast_to(
        np.eye(pd, dtype=np.complex128) * (float(np.mean(r_diag)) + 1e-6),
        (ntiles, pd, pd),
    ).copy()

    for t in range(p, t_len):
        z_pred = z_hat @ f_aug.T
        p_pred = f_aug[None, :, :] @ p_hat @ f_aug.conj().T[None, :, :] + q_aug[None, :, :]

        innov = y_hist_batch[t].astype(np.complex128) - z_pred[:, :d]
        s = p_pred[:, :d, :d] + r_mat[None, :, :]
        k = p_pred[:, :, :d] @ np.linalg.pinv(s)

        z_hat = z_pred + (k @ innov[:, :, None]).squeeze(-1)
        p_hat = p_pred - (k @ p_pred[:, :d, :])

    z_next = z_hat @ f_aug.T
    return z_next[:, :d]


def steady_kalman_predict_next_batch(
    y_hist_batch: np.ndarray,
    f_aug: np.ndarray,
    k_ss: np.ndarray,
) -> np.ndarray:
    t_len, _, d = y_hist_batch.shape
    p = f_aug.shape[0] // d
    if t_len <= p:
        return y_hist_batch[-1].astype(np.complex128)

    z_hat = _init_augmented_state_batch(y_hist_batch, p)

    for t in range(p, t_len):
        z_pred = z_hat @ f_aug.T
        innov = y_hist_batch[t].astype(np.complex128) - z_pred[:, :d]
        z_hat = z_pred + (k_ss[None, :, :] @ innov[:, :, None]).squeeze(-1)

    z_next = z_hat @ f_aug.T
    return z_next[:, :d]


class ConfiguredWeightsESN:
    def __init__(
        self,
        poles: np.ndarray,
        residues: np.ndarray,
        d_out: int,
        d_in: int,
        activation: str = "identity",
        spectral_radius: float = 0.3,
        input_scale: float = 0.15,
    ):
        # poles: [M, K], residues: [M, d_out*d_in, K]
        self.poles = poles.astype(np.complex128)
        self.residues = residues.astype(np.complex128)
        self.num_basis = residues.shape[0]
        self.num_terms = residues.shape[2]
        self.d_out = d_out
        self.d_in = d_in
        self.activation = activation
        pole_mags = np.abs(self.poles)
        pole_scales = np.where(
            pole_mags > float(spectral_radius),
            float(spectral_radius) / (pole_mags + 1e-12),
            1.0,
        )
        self.poles = self.poles * pole_scales
        self.W_res = self.poles
        self.W_in = np.transpose(self.residues, (0, 2, 1)).reshape(
            self.num_basis,
            self.num_terms,
            self.d_out,
            self.d_in,
            order="F",
        )
        self.W_in = self.W_in * float(input_scale)
        self.state = np.zeros((self.num_basis, self.num_terms, d_out), dtype=np.complex128)

    def reset(self):
        self.state.fill(0.0)

    def step(self, u_t: np.ndarray) -> np.ndarray:
        driven = (self.W_in @ u_t[..., None]).squeeze(-1)
        pre_act = self.W_res[..., None] * self.state + driven
        self.state = apply_complex_activation(pre_act, self.activation)
        return self.state

def apply_complex_activation(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "identity":
        return x
    if activation == "tanh":
        return np.tanh(np.real(x)) + 1j * np.tanh(np.imag(x))
    if activation == "relu":
        return np.maximum(np.real(x), 0.0) + 1j * np.maximum(np.imag(x), 0.0)
    raise ValueError(f"Unsupported activation: {activation}")

class RandomWeightsESN:
    def __init__(
        self,
        num_basis: int,
        num_terms: int,
        d_out: int,
        d_in: int,
        rng: np.random.Generator,
        activation: str = "identity",
        spectral_radius: float = 0.5,
        input_scale: float = 0.15,
    ):
        self.num_basis = num_basis
        self.num_terms = num_terms
        self.d_out = d_out
        self.d_in = d_in
        self.activation = activation
        pole_mags = rng.uniform(0.0, 1.0, size=(num_basis, num_terms))
        pole_phases = rng.uniform(-np.pi, np.pi, size=(num_basis, num_terms))
        self.poles = pole_mags * np.exp(1j * pole_phases)
        curr_radius = float(np.max(np.abs(self.poles)))
        if np.isfinite(curr_radius) and curr_radius > 1e-12:
            self.poles = self.poles * (float(spectral_radius) / curr_radius)
        self.W_res = self.poles
        self.W_in = input_scale * (
            rng.standard_normal((num_basis, num_terms, d_out, d_in))
            + 1j * rng.standard_normal((num_basis, num_terms, d_out, d_in))
        ) / np.sqrt(2.0 * max(d_in, 1))
        self.state = np.zeros((num_basis, num_terms, d_out), dtype=np.complex128)

    def reset(self):
        self.state.fill(0.0)

    def step(self, u_t: np.ndarray) -> np.ndarray:
        driven = (self.W_in @ u_t[..., None]).squeeze(-1)
        pre_act = self.W_res[..., None] * self.state + driven
        self.state = apply_complex_activation(pre_act, self.activation)
        return self.state



def collect_esn_states_per_tile(esn, y_hist_chunk: np.ndarray) -> np.ndarray:
    # y_hist_chunk: [T, Ntiles, D]
    t_len, ntiles, _ = y_hist_chunk.shape
    feat_dim = esn.num_basis * esn.num_terms * esn.d_out
    feats = np.zeros((t_len, ntiles, feat_dim), dtype=np.complex128)
    W_in = esn.W_in.astype(np.complex128)
    W_res = esn.W_res.astype(np.complex128)
    state = np.zeros((ntiles, esn.num_basis, esn.num_terms, esn.d_out), dtype=np.complex128)

    for t in range(t_len):
        u_t = y_hist_chunk[t].astype(np.complex128)
        driven = (W_in[None, :, :, :, :] @ u_t[:, None, None, :, None]).squeeze(-1)
        pre_act = W_res[None, :, :, None] * state + driven
        state = apply_complex_activation(pre_act, esn.activation)
        feats[t] = state.reshape(ntiles, feat_dim)
    return feats


def ls_readout_train_predict_next(
    feat_hist: np.ndarray,
    target_hist: np.ndarray,
    target_next: np.ndarray,
    reg: float = 1e-6,
) -> np.ndarray:
    # feat_hist: [T, Ntiles, F], target_hist: [T, Ntiles, D], target_next: [Ntiles, D]
    z_train = feat_hist[:-1].reshape(-1, feat_hist.shape[-1])
    y_train = target_hist[1:].reshape(-1, target_hist.shape[-1])
    z_test = feat_hist[-1]

    # Ridge-regularized LS solve for Z @ W ~= Y:
    # W = (Z^H Z + λI)^(-1) Z^H Y
    f_dim = z_train.shape[1]
    gram = z_train.conj().T @ z_train + reg * np.eye(f_dim, dtype=np.complex128)
    gain = np.linalg.pinv(gram) @ z_train.conj().T
    W_out = gain @ y_train
    _ = target_next
    return z_test @ W_out


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


def frequency_matrix_to_q_column(qf: np.ndarray) -> np.ndarray:
    return qf.reshape(-1)


def fit_shared_denominator_vector_rational(
    q_col: np.ndarray, num_freqs: int, value_dim: int, degree: int, omegas: np.ndarray
):
    qf = q_column_to_frequency_matrix(q_col, num_freqs, value_dim)
    zinv = np.exp(-1j * omegas)
    z = np.stack([zinv ** k for k in range(1, degree + 1)], axis=1)
    num_rows = num_freqs * value_dim
    q_col = qf.reshape(num_rows)
    z_rep = np.repeat(z, value_dim, axis=0)

    # Sparse LS system:
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


def build_configured_esn_from_kalman_stats(
    tiles_noisy: np.ndarray,
    ar_order: int,
    history_len: int,
    r_diag: np.ndarray,
    num_basis: int,
    degree: int,
    num_freqs: int,
    activation: str,
    diagnostics: dict | None = None,
) -> ConfiguredWeightsESN:
    t_dec, _, d = tiles_noisy.shape
    p_eff = min(ar_order, history_len - 1)
    kf_helper = kalman_filter_pred(ar_order=ar_order)
    omegas = np.linspace(0.0, np.pi, num_freqs, endpoint=True)

    v_list = []
    for s in range(0, t_dec - history_len):
        y_hist_chunk = tiles_noisy[s : s + history_len]
        a_blocks, q_proc = kf_helper._estimate_ar_p_q_joint(y_hist_chunk, p_eff)
        a_blocks = [a.conj() for a in a_blocks]
        f_aug, q_aug = kf_helper._build_augmented_system(a_blocks, q_proc)
        h_mat = build_augmented_obs_matrix(d, p_eff)
        r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex128))
        _, k_ss = solve_riccati_steady_state_complex(f_aug, q_aug, h_mat, r_mat)
        h_samps = steady_state_predictor_transfer_samples_from_kalman(f_aug, k_ss, d=d, num_freqs=num_freqs)
        v_list.append(vectorize_transfer_samples(h_samps))

    v = np.stack(v_list, axis=1)
    mean_v, vc, _ = estimate_empirical_complex_covariance(v, compute_kvv=False)
    m = min(num_basis, vc.shape[0], vc.shape[1])
    m = num_basis
    if m < 1:
        raise ValueError("num_basis must be at least 1.")

    # Compute only the dominant modes: use a skinny-SVD path when V is tall,
    # otherwise use a partial Hermitian eigensolver on the symmetrized covariance.
    if vc.shape[0] > vc.shape[1]:
        u, svals, _ = np.linalg.svd(vc, full_matrices=False)
        evals = (svals**2) / max(vc.shape[1], 1)
        q_eig = u[:, :m]
    else:
        kvv = (vc @ vc.conj().T) / max(vc.shape[1], 1)
        kvv = 0.5 * (kvv + kvv.conj().T)
        if m < kvv.shape[0]:
            evals, q_eig = eigsh(kvv, k=m, which="LA")
        else:
            evals, evecs = np.linalg.eigh(kvv)
            q_eig = evecs
    idx = np.argsort(evals)[::-1][:m]
    evals_sorted = np.sort(np.real(evals))[::-1]
    q_eig = q_eig[:, idx]
    assert q_eig.shape == (num_freqs * d * d, m), "Leading eigenvector matrix shape changed unexpectedly."

    value_dim = d * d
    poles_all = np.zeros((m, degree), dtype=np.complex128)
    residues_all = np.zeros((m, value_dim, degree), dtype=np.complex128)
    for j in range(m):
        q_col = q_eig[:, j]
        assert q_col.shape == (num_freqs * value_dim,), "q_col shape changed unexpectedly."
        assert q_column_to_frequency_matrix(q_col, num_freqs, value_dim).shape == (
            num_freqs,
            value_dim,
        ), "Downstream q_col reshape changed unexpectedly."
        fit = fit_shared_denominator_vector_rational(
            q_col=q_col, num_freqs=num_freqs, value_dim=value_dim, degree=degree, omegas=omegas
        )
        poles, residues = decompose_rp_fit_into_first_order(
            fit=fit, q_col=q_col, num_freqs=num_freqs, value_dim=value_dim, omegas=omegas
        )
        poles_all[j] = poles
        residues_all[j] = residues
    _ = mean_v
    if diagnostics is not None:
        total_eval = float(np.sum(np.maximum(evals_sorted, 0.0)))
        cum_energy = (
            np.cumsum(np.maximum(evals_sorted, 0.0)) / max(total_eval, 1e-15)
            if evals_sorted.size > 0
            else np.asarray([], dtype=np.float64)
        )
        energy_thresholds = [0.90, 0.95, 0.99]
        suggested_m = {}
        for thr in energy_thresholds:
            idx_thr = int(np.searchsorted(cum_energy, thr, side="left")) + 1 if cum_energy.size > 0 else 0
            suggested_m[thr] = min(idx_thr, int(evals_sorted.size))
        pole_mag = np.abs(poles_all.reshape(-1))
        residue_mag = np.abs(residues_all.reshape(-1))
        diagnostics.update(
            {
                "kvv_eigenvalues_sorted": evals_sorted,
                "kvv_cumulative_energy": cum_energy,
                "suggested_m": suggested_m,
                "configured_pole_mag_min": float(np.min(pole_mag)) if pole_mag.size > 0 else 0.0,
                "configured_pole_mag_p50": float(np.quantile(pole_mag, 0.50)) if pole_mag.size > 0 else 0.0,
                "configured_pole_mag_p90": float(np.quantile(pole_mag, 0.90)) if pole_mag.size > 0 else 0.0,
                "configured_pole_mag_max": float(np.max(pole_mag)) if pole_mag.size > 0 else 0.0,
                "configured_residue_mag_p50": float(np.quantile(residue_mag, 0.50)) if residue_mag.size > 0 else 0.0,
                "configured_residue_mag_p90": float(np.quantile(residue_mag, 0.90)) if residue_mag.size > 0 else 0.0,
                "configured_residue_mag_max": float(np.max(residue_mag)) if residue_mag.size > 0 else 0.0,
                "num_transfer_samples": int(v.shape[1]),
                "transfer_feature_dim": int(v.shape[0]),
            }
        )
    return ConfiguredWeightsESN(poles=poles_all, residues=residues_all, d_out=d, d_in=d, activation=activation)


def build_random_esn(d: int, num_basis: int, degree: int, seed: int, activation: str):
    rng = np.random.default_rng(seed)
    return RandomWeightsESN(
        num_basis=num_basis,
        num_terms=degree,
        d_out=d,
        d_in=d,
        rng=rng,
        activation=activation,
        spectral_radius=0.5,
        input_scale=0.8,
    )

def evaluate_nmse_over_chunks(
    h_clean_dec: np.ndarray,
    snr_db: float,
    history_len: int,
    ar_order: int,
    num_basis: int,
    rp_degree: int,
    num_freqs: int,
    activation: str,
    ls_reg: float,
    seed: int,
    offline_ratio: float = 0.5,
    run_diagnostics: bool = False,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)

    h_noisy_dec, noise_var = add_complex_awgn(h_clean_dec, snr_db, rng)
    tiles_clean = channels_to_tiles(h_clean_dec)  # [T, Ntiles, D]
    tiles_noisy = channels_to_tiles(h_noisy_dec)

    t_dec, _, d = tiles_clean.shape
    if t_dec <= history_len:
        raise ValueError("Not enough decimated samples for at least one chunk.")
    if not (0.0 < float(offline_ratio) <= 1.0):
        raise ValueError(f"offline_ratio must be in (0, 1], got {offline_ratio}.")

    if float(offline_ratio) >= 1.0:
        # Special case requested by user:
        # configure on all data and evaluate on all data (original behavior).
        offline_len = t_dec
        online_len = t_dec
        tiles_clean_offline = tiles_clean
        tiles_noisy_offline = tiles_noisy
        tiles_clean_online = tiles_clean
        tiles_noisy_online = tiles_noisy
    else:
        offline_len = int(np.floor(t_dec * float(offline_ratio)))
        offline_len = max(1, min(offline_len, t_dec - 1))
        online_len = t_dec - offline_len
        if offline_len <= history_len:
            raise ValueError(
                f"Offline phase too short for statistics collection: offline_len={offline_len}, history_len={history_len}."
            )
        if online_len <= history_len:
            raise ValueError(
                f"Online phase too short for prediction evaluation: online_len={online_len}, history_len={history_len}."
            )

        tiles_clean_offline = tiles_clean[:offline_len]
        tiles_noisy_offline = tiles_noisy[:offline_len]
        tiles_clean_online = tiles_clean[offline_len:]
        tiles_noisy_online = tiles_noisy[offline_len:]

    kf_helper = kalman_filter_pred(ar_order=ar_order)
    p_eff = min(ar_order, history_len - 1)

    # shared R across all tiles/subcarriers
    r_diag = noise_var * np.ones((d,), dtype=np.float64)
    diag_info = {} if run_diagnostics else None
    configured_esn = build_configured_esn_from_kalman_stats(
        tiles_noisy=tiles_noisy_offline,
        ar_order=ar_order,
        history_len=history_len,
        r_diag=r_diag,
        num_basis=num_basis,
        degree=rp_degree,
        num_freqs=num_freqs,
        activation=activation,
        diagnostics=diag_info,
    )
    random_esn = build_random_esn(
        d=d,
        num_basis=num_basis,
        degree=rp_degree,
        seed=seed,
        activation=activation,
    )

    num_full = 0.0
    den_full = 0.0
    num_ss = 0.0
    den_ss = 0.0
    num_cfg = 0.0
    den_cfg = 0.0
    num_rand = 0.0
    den_rand = 0.0

    cond_cfg_vals = []
    cond_rand_vals = []

    for s in range(0, online_len - history_len):
        y_hist_chunk = tiles_noisy_online[s : s + history_len]       # [history_len, Ntiles, D]
        x_hist_chunk = tiles_clean_online[s : s + history_len]       # [history_len, Ntiles, D]
        x_true_next = tiles_clean_online[s + history_len]            # [Ntiles, D]

        a_blocks, q_proc = kf_helper._estimate_ar_p_q_joint(y_hist_chunk, p_eff)
        a_blocks = [a.conj() for a in a_blocks]
        f_aug, q_aug = kf_helper._build_augmented_system(a_blocks, q_proc)

        h_mat = build_augmented_obs_matrix(d, p_eff)
        r_mat = np.diag(np.maximum(r_diag, 1e-12).astype(np.complex128))
        _, k_ss = solve_riccati_steady_state_complex(f_aug, q_aug, h_mat, r_mat)

        x_pred_full = full_kalman_predict_next_batch(y_hist_chunk, f_aug, q_aug, r_diag)
        x_pred_ss = steady_kalman_predict_next_batch(y_hist_chunk, f_aug, k_ss)
        esn_states_cfg = collect_esn_states_per_tile(configured_esn, y_hist_chunk)
        esn_states_rand = collect_esn_states_per_tile(random_esn, y_hist_chunk)
        x_pred_cfg = ls_readout_train_predict_next(
            esn_states_cfg, y_hist_chunk, x_true_next, reg=ls_reg
        )
        x_pred_rand = ls_readout_train_predict_next(
            esn_states_rand, y_hist_chunk, x_true_next, reg=ls_reg
        )

        if run_diagnostics:
            z_cfg = esn_states_cfg[:-1].reshape(-1, esn_states_cfg.shape[-1])
            z_rand = esn_states_rand[:-1].reshape(-1, esn_states_rand.shape[-1])
            gram_cfg = z_cfg.conj().T @ z_cfg + ls_reg * np.eye(z_cfg.shape[1], dtype=np.complex128)
            gram_rand = z_rand.conj().T @ z_rand + ls_reg * np.eye(z_rand.shape[1], dtype=np.complex128)
            cond_cfg_vals.append(float(np.linalg.cond(gram_cfg)))
            cond_rand_vals.append(float(np.linalg.cond(gram_rand)))

        num_full += float(np.sum(np.abs(x_pred_full - x_true_next) ** 2))
        den_full += float(np.sum(np.abs(x_true_next) ** 2))

        num_ss += float(np.sum(np.abs(x_pred_ss - x_true_next) ** 2))
        den_ss += float(np.sum(np.abs(x_true_next) ** 2))

        num_cfg += float(np.sum(np.abs(x_pred_cfg - x_true_next) ** 2))
        den_cfg += float(np.sum(np.abs(x_true_next) ** 2))
        
        num_rand += float(np.sum(np.abs(x_pred_rand - x_true_next) ** 2))
        den_rand += float(np.sum(np.abs(x_true_next) ** 2))

    nmse_full = num_full / max(den_full, 1e-15)
    nmse_ss = num_ss / max(den_ss, 1e-15)
    nmse_cfg = num_cfg / max(den_cfg, 1e-15)
    nmse_rand = num_rand / max(den_rand, 1e-15)

    if run_diagnostics and diag_info is not None:
        def _fmt_float(x: float) -> str:
            return f"{x:.3e}"

        print("\n[ESN diagnostics]")
        print(
            "K_vv eigvals (top-10): "
            + ", ".join(_fmt_float(float(vv)) for vv in diag_info["kvv_eigenvalues_sorted"][:10])
        )
        cum_e = diag_info["kvv_cumulative_energy"]
        print(
            "K_vv cumulative energy (top-10 modes): "
            + ", ".join(_fmt_float(float(cc)) for cc in cum_e[:10])
        )
        sugg_m = diag_info["suggested_m"]
        print(
            f"Suggested esn_m by explained-energy thresholds: "
            f"90%->{sugg_m[0.90]}, 95%->{sugg_m[0.95]}, 99%->{sugg_m[0.99]}"
        )
        print(
            f"Configured pole |.| stats: min={diag_info['configured_pole_mag_min']:.3f}, "
            f"p50={diag_info['configured_pole_mag_p50']:.3f}, "
            f"p90={diag_info['configured_pole_mag_p90']:.3f}, "
            f"max={diag_info['configured_pole_mag_max']:.3f}"
        )
        print(
            f"Configured residue |.| stats: p50={diag_info['configured_residue_mag_p50']:.3e}, "
            f"p90={diag_info['configured_residue_mag_p90']:.3e}, "
            f"max={diag_info['configured_residue_mag_max']:.3e}"
        )
        if len(cond_cfg_vals) > 0 and len(cond_rand_vals) > 0:
            cfg_cond = np.asarray(cond_cfg_vals)
            rand_cond = np.asarray(cond_rand_vals)
            print(
                f"LS Gram cond (configured): median={np.median(cfg_cond):.3e}, "
                f"p90={np.quantile(cfg_cond, 0.90):.3e}, max={np.max(cfg_cond):.3e}"
            )
            print(
                f"LS Gram cond (random): median={np.median(rand_cond):.3e}, "
                f"p90={np.quantile(rand_cond, 0.90):.3e}, max={np.max(rand_cond):.3e}"
            )
        print(
            f"Transfer-stat sample count={diag_info['num_transfer_samples']}, "
            f"feature_dim={diag_info['transfer_feature_dim']}"
        )
        print("[/ESN diagnostics]\n")

    return nmse_ss, nmse_full, nmse_cfg, nmse_rand


def main():
    parser = argparse.ArgumentParser(description="P2P Kalman channel prediction NMSE vs SNR on ns3 saved channels")
    parser.add_argument("--ns3-root", type=str, default="ns3", help="Root folder containing channels_<mobility>_<drop>")
    parser.add_argument("--mobility", type=str, default="higher_mobility", help="Mobility folder key")
    parser.add_argument("--drop-idx", type=int, default=1)
    parser.add_argument("--start-slot", type=int, default=1)
    parser.add_argument("--end-slot", type=int, default=100)
    parser.add_argument("--feedback-delay", type=int, default=4)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--ar-order", type=int, default=2)
    parser.add_argument("--esn-m", "--fb-m", dest="esn_m", type=int, default=4, help="Number of configured/random ESN basis vectors")
    parser.add_argument("--esn-k", "--fb-k", dest="esn_k", type=int, default=4, help="Rational polynomial degree and ESN modal terms")
    parser.add_argument("--esn-num-freqs", "--fb-num-freqs", dest="esn_num_freqs", type=int, default=64, help="Frequency samples for transfer statistics")
    parser.add_argument(
        "--esn-activation", "--fb-activation",
        dest="esn_activation",
        type=str,
        default="tanh",
        choices=["identity", "tanh", "relu"],
        help="Activation used in configured/random ESN reservoirs",
    )
    parser.add_argument(
        "--esn-ls-reg", "--fb-ls-reg",
        dest="esn_ls_reg",
        type=float,
        default=1e-6,
        help="Ridge regularization used by ESN readout solve",
    )
    parser.add_argument("--rx-ant", type=int, default=4)
    parser.add_argument("--tx-ant", type=int, default=4)
    parser.add_argument("--snr-start", type=int, default=0)
    parser.add_argument("--snr-stop", type=int, default=15)
    parser.add_argument("--snr-step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--offline-ratio",
        type=float,
        default=0.5,
        help="Fraction in (0,1] of decimated slots used for offline WESN configuration; 1.0 uses all data for both configuration and testing.",
    )
    parser.add_argument("--plot-path", type=str, default="results/kalman_p2p_nmse_vs_snr.png")
    parser.add_argument(
        "--esn-diagnostics",
        action="store_true",
        default=False,
        help="Print diagnostics for configured-ESN hyperparameter selection (K_vv spectrum, pole/residue stats, LS conditioning).",
    )
    args = parser.parse_args()

    save_path = f"results/kalman_p2p_nmse_vs_snr_data_activation_{args.esn_activation}_{args.mobility}.npz"

    h_clean_dec, selected_slots = load_clean_p2p_channels(
        ns3_folder=Path(args.ns3_root),
        drop_idx=args.drop_idx,
        mobility=args.mobility,
        start_slot=args.start_slot,
        end_slot=args.end_slot,
        feedback_delay=args.feedback_delay,
        rx_ant=args.rx_ant,
        tx_ant=args.tx_ant,
    )

    print(f"Loaded decimated slots ({len(selected_slots)}): {selected_slots[:10]}{' ...' if len(selected_slots) > 10 else ''}")
    print(f"Clean decimated channel tensor shape: {h_clean_dec.shape}")

    snr_vals = np.arange(args.snr_start, args.snr_stop + 1, args.snr_step)
    nmse_ss_vals = []
    nmse_full_vals = []
    nmse_cfg_vals = []
    nmse_rand_vals = []

    for snr_db in snr_vals:
        nmse_ss, nmse_full, nmse_cfg, nmse_rand = evaluate_nmse_over_chunks(
            h_clean_dec=h_clean_dec,
            snr_db=float(snr_db),
            history_len=args.history_len,
            ar_order=args.ar_order,
            num_basis=args.esn_m,
            rp_degree=args.esn_k,
            num_freqs=args.esn_num_freqs,
            activation=args.esn_activation,
            ls_reg=float(args.esn_ls_reg),
            seed=args.seed,
            offline_ratio=float(args.offline_ratio),
            run_diagnostics=bool(args.esn_diagnostics),
        )
        nmse_ss_vals.append(nmse_ss)
        nmse_full_vals.append(nmse_full)
        nmse_cfg_vals.append(nmse_cfg)
        nmse_rand_vals.append(nmse_rand)
        print(
            f"SNR={snr_db:>2d} dB | NMSE steady={nmse_ss:.4e}, full={nmse_full:.4e}, "
            f"configured_esn={nmse_cfg:.4e}, random_esn={nmse_rand:.4e}"
        )

        hold = 1

    nmse_ss_vals = np.asarray(nmse_ss_vals)
    nmse_full_vals = np.asarray(nmse_full_vals)
    nmse_cfg_vals = np.asarray(nmse_cfg_vals)
    nmse_rand_vals = np.asarray(nmse_rand_vals)

    out_path = Path(args.plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_path.with_name(
        out_path.stem + f"_activation_{args.esn_activation}_{args.mobility}" + out_path.suffix
    )

    plt.figure(figsize=(8, 5))
    plt.plot(snr_vals, nmse_ss_vals, marker="o", label="Steady-state Kalman")
    plt.plot(snr_vals, nmse_full_vals, marker="s", label="Full Kalman")
    plt.plot(snr_vals, nmse_cfg_vals, marker="^", label="Configured ESN + LS readout")
    plt.plot(snr_vals, nmse_rand_vals, marker="d", label="Random ESN + LS readout")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Channel prediction NMSE")
    plt.title(
        "P2P channel prediction NMSE vs SNR\n"
        f"({args.mobility}, drop={args.drop_idx}, delay={args.feedback_delay}, history={args.history_len}, AR={args.ar_order})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # plt.ylim(bottom=0, top=1.0)
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")

    np.savez(
        save_path,
        snr_vals=snr_vals,
        nmse_ss_vals=nmse_ss_vals,
        nmse_full_vals=nmse_full_vals,
        nmse_cfg_vals=nmse_cfg_vals,
        nmse_rand_vals=nmse_rand_vals,
    )
    print(f"Saved raw NMSE data to: {save_path}")


if __name__ == "__main__":
    main()