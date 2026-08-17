import argparse
from pathlib import Path
import re
import sys
import os
import types
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)

# Avoid importing the full TensorFlow/PyTorch simulation stack; this standalone
# experiment needs only the NumPy/SciPy Kalman predictor module.
if "dmimo" not in sys.modules:
    dmimo_package = types.ModuleType("dmimo")
    dmimo_package.__path__ = [os.path.join(dmimo_root, "dmimo")]
    sys.modules["dmimo"] = dmimo_package
if "dmimo.channel" not in sys.modules:
    channel_package = types.ModuleType("dmimo.channel")
    channel_package.__path__ = [os.path.join(dmimo_root, "dmimo", "channel")]
    sys.modules["dmimo.channel"] = channel_package

from dmimo.channel.kalman_filter_pred import kalman_filter_pred
from dmimo.channel.configured_wesn_pred import (
    configured_wesn_pred,
    _freeze_predictor_weights,
)
from dmimo.channel.steady_state_kalman_filter_pred import (
    steady_state_kalman_filter_pred,
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
    "savefig.dpi": 300,
})


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
            if "Hdm" in data:
                # [all_rx_ant, all_tx_ant, num_syms, num_sc]
                hdm = data["Hdm"]
            elif "h_freq_csi" in data:
                # Saved estimator format:
                # [batch, rx_node, rx_ant, tx_node, tx_ant, sym, sc]
                saved_csi = np.asarray(data["h_freq_csi"])
                if saved_csi.ndim != 7:
                    raise ValueError(
                        f"Unexpected h_freq_csi shape {saved_csi.shape} in {by_slot[s]}"
                    )
                hdm = saved_csi[0, 0, :, 0, :, :, :]
            else:
                raise KeyError(
                    f"Expected 'Hdm' or 'h_freq_csi' in {by_slot[s]}; "
                    f"found {data.files}."
                )
        h_p2p = hdm[:rx_ant, :tx_ant, :, :].astype(np.complex128)
        h_list.append(h_p2p)

    # [T_decimated, rx_ant, tx_ant, num_syms, num_sc]
    h_seq = np.stack(h_list, axis=0)
    return h_seq, np.asarray(selected_slots, dtype=int)


def add_complex_awgn(h: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Add temporally white circular AWGN with variance E[|h|^2]/10^(snr_db/10).

    Used for both observation noise v_t (R) and extra process noise w_t (Q);
    the caller decides whether the noisy tensor is the observation or the true state.
    """
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


def tiles_to_csi_history(tiles: np.ndarray, num_rx: int, num_tx: int) -> np.ndarray:
    """Pack independent P2P tiles into the 8D history configured WESN expects.

    Each (symbol, subcarrier) tile is stored as its own frequency bin with a
    single OFDM symbol so resource-block averaging with width 1 is a no-op.

    tiles layout is [T, Ntiles, rx*tx] with C-order (rx, tx). A raw reshape into
    [T, 1, 1, rx, 1, tx, 1, Ntiles] would mix tile and antenna axes.
    """
    t_len, num_tiles, dim = tiles.shape
    if dim != num_rx * num_tx:
        raise ValueError(
            f"Tile dimension {dim} does not match {num_rx}x{num_tx} antennas."
        )
    antennas = tiles.reshape(t_len, num_tiles, num_rx, num_tx)
    # [T, rx, tx, Ntiles] then singleton batch/node/OFDM axes.
    packed = np.transpose(antennas, (0, 2, 3, 1))
    return packed[:, None, None, :, None, :, None, :]


def csi_prediction_to_tiles(pred: np.ndarray, num_rx: int, num_tx: int) -> np.ndarray:
    """Undo tiles_to_csi_history() for a single predicted slot."""
    if pred.ndim != 7:
        raise ValueError(f"Expected 7D configured-WESN prediction, got {pred.shape}.")
    packed = pred[0, 0, :num_rx, 0, :num_tx, 0, :]
    num_tiles = packed.shape[-1]
    return np.transpose(packed, (2, 0, 1)).reshape(num_tiles, num_rx * num_tx)


def channel_seq_to_history(h_seq: np.ndarray) -> np.ndarray:
    """[T, rx, tx, sym, sc] -> production 8D CSI history."""
    if h_seq.ndim != 5:
        raise ValueError(f"Expected [T, rx, tx, sym, sc], got {h_seq.shape}.")
    return h_seq[:, None, None, :, None, :, :, :]


def make_wesn_rc_config(
    *,
    esn_m: int,
    esn_k: int,
    num_freqs: int,
    activation: str,
    ls_reg: float,
    hankel_energy: float,
    input_scale: float,
    window_length: int,
    subcarriers_per_rb: int,
    lite: bool,
    random: bool,
):
    """Match sims/sim_mu_mimo_testing_updates.py / twc_tput_across_tx.sh."""
    is_balanced = not random
    return types.SimpleNamespace(
        W_tran_sparsity=0.4,
        W_tran_radius=1.0 if is_balanced else 0.5,
        input_scale=float(input_scale),
        window_length=int(window_length),
        regularization=float(ls_reg),
        enable_window=True,
        enable_kalman_weight_config=not random,
        kalman_eps=1e-8,
        esn_m=int(esn_m),
        esn_k=int(esn_k),
        esn_num_freqs=int(num_freqs),
        esn_activation=str(activation),
        esn_ls_reg=float(ls_reg),
        esn_diagnostics=False,
        enable_skip_connections=True,
        wesn_online_update="batch_ridge",
        enable_residue_low_rank=False,
        enable_balanced_truncation=is_balanced,
        enable_balanced_hankel_truncation=bool(lite) and is_balanced,
        balanced_hankel_energy_threshold=float(hankel_energy),
        residue_energy_threshold=0.95,
        reservoir_readout_regularization=1e-2,
        skip_readout_regularization=float(ls_reg),
        wesn_lite_readout_mode="centered_ridge",
        wesn_lite_subcarriers_per_rb=int(subcarriers_per_rb),
    )


def balanced_wesn_rank_summary(predictor: configured_wesn_pred) -> dict:
    metrics = predictor.predictor_complexity_metrics
    orders = np.asarray(
        metrics.get("balanced_orders_per_mode", [predictor.state_dim]),
        dtype=np.int64,
    )
    if orders.size == 0:
        orders = np.asarray([predictor.state_dim], dtype=np.int64)
    unique, counts = np.unique(orders, return_counts=True)
    retained = np.asarray(
        metrics.get("balanced_retained_hankel_energy_per_mode", [np.nan]),
        dtype=np.float64,
    )
    return {
        "threshold": float(metrics.get("balanced_hankel_energy_threshold", np.nan)),
        "mean": float(np.mean(orders)),
        "median": float(np.median(orders)),
        "mode": int(unique[np.argmax(counts)]),
        "min": int(np.min(orders)),
        "max": int(np.max(orders)),
        "state_dimension": int(predictor.state_dim),
        "retained_energy_mean": float(np.nanmean(retained)),
        "method": str(metrics.get("method", "configured_wesn_balanced")),
    }


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
        self.raw_poles = poles.astype(np.complex128)
        self.poles = self.raw_poles.copy()
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


class LowRankConfiguredWeightsESN:
    """Low-rank configured reservoir built from the pole residues.

    For each residue C = U Sigma V^H, retain the smallest rank containing the
    requested Frobenius energy and realize

        x[t] = p*x[t-1] + Sigma_r*V_r^H*u[t],
        y[t] = U_r*x[t].

    With identity activation this is the direct truncated linear realization.
    With tanh it uses the same configured factors as nonlinear reservoir
    features, followed by the selected trained readout.
    """

    def __init__(
        self,
        poles: np.ndarray,
        residues: np.ndarray,
        d_out: int,
        d_in: int,
        energy_threshold: float = 0.95,
        activation: str = "tanh",
    ):
        if d_out != d_in:
            raise ValueError("WESN-Lite currently expects square D-by-D residues.")
        if not (0.0 < float(energy_threshold) <= 1.0):
            raise ValueError("energy_threshold must be in (0, 1].")

        recurrent_weights = []
        input_rows = []
        output_columns = []
        ranks = []
        retained_energies = []

        for mode_idx in range(poles.shape[0]):
            for pole_idx in range(poles.shape[1]):
                # vectorize_transfer_samples() stores vec(C) as C.T.reshape(-1).
                residue = residues[mode_idx, :, pole_idx].reshape(d_in, d_out).T
                u, singular_values, vh = np.linalg.svd(
                    residue, full_matrices=False
                )
                energy = singular_values**2
                total_energy = float(np.sum(energy))
                if total_energy <= np.finfo(np.float64).eps:
                    rank = 1
                    retained = 1.0
                else:
                    cumulative = np.cumsum(energy) / total_energy
                    rank = int(
                        np.searchsorted(cumulative, energy_threshold, side="left")
                    ) + 1
                    rank = min(max(rank, 1), singular_values.size)
                    retained = float(cumulative[rank - 1])

                ranks.append(rank)
                retained_energies.append(retained)
                for factor_idx in range(rank):
                    recurrent_weights.append(poles[mode_idx, pole_idx])
                    input_rows.append(
                        singular_values[factor_idx] * vh[factor_idx]
                    )
                    output_columns.append(u[:, factor_idx])

        self.W_res = np.asarray(recurrent_weights, dtype=np.complex128)
        self.W_in = np.asarray(input_rows, dtype=np.complex128)
        self.W_out_reference = np.stack(output_columns, axis=1).astype(
            np.complex128
        )
        self.state_dim = int(self.W_res.size)
        self.d_out = int(d_out)
        self.d_in = int(d_in)
        self.ranks = np.asarray(ranks, dtype=np.int64)
        self.retained_energies = np.asarray(
            retained_energies, dtype=np.float64
        )
        self.energy_threshold = float(energy_threshold)
        self.activation = str(activation)

    def rank_summary(self) -> dict:
        unique, counts = np.unique(self.ranks, return_counts=True)
        return {
            "threshold": self.energy_threshold,
            "retained_energy_min": float(np.min(self.retained_energies)),
            "retained_energy_mean": float(np.mean(self.retained_energies)),
            "mean": float(np.mean(self.ranks)),
            "median": float(np.median(self.ranks)),
            "mode": int(unique[np.argmax(counts)]),
            "min": int(np.min(self.ranks)),
            "max": int(np.max(self.ranks)),
            "state_dimension": self.state_dim,
        }

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


def collect_low_rank_esn_states_per_tile(
    esn: LowRankConfiguredWeightsESN,
    y_hist_chunk: np.ndarray,
) -> np.ndarray:
    """Roll out the low-rank configured reservoir for every channel tile."""
    t_len, ntiles, _ = y_hist_chunk.shape
    features = np.zeros(
        (t_len, ntiles, esn.state_dim), dtype=np.complex128
    )
    state = np.zeros((ntiles, esn.state_dim), dtype=np.complex128)
    for t in range(t_len):
        driven = y_hist_chunk[t].astype(np.complex128) @ esn.W_in.T
        preactivation = state * esn.W_res[None, :] + driven
        state = apply_complex_activation(preactivation, esn.activation)
        features[t] = state
    return features


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


def centered_ridge_readout_train_predict_next(
    feat_hist: np.ndarray,
    target_hist: np.ndarray,
    reference_readout: np.ndarray,
    reg: float = 1e-2,
) -> np.ndarray:
    """Fit a full readout while penalizing departure from its configured value."""
    z_train = feat_hist[:-1].reshape(-1, feat_hist.shape[-1])
    y_train = target_hist[1:].reshape(-1, target_hist.shape[-1])
    z_test = feat_hist[-1]
    if reference_readout.shape != (y_train.shape[1], z_train.shape[1]):
        raise ValueError(
            "reference_readout must have shape [output_dim, feature_dim]."
        )

    # min_B ||Y-ZB||_F^2/N + reg*||B-B_ref||_F^2
    num_samples = z_train.shape[0]
    reg_normal_eq = num_samples * float(reg)
    gram = z_train.conj().T @ z_train + reg_normal_eq * np.eye(
        z_train.shape[1], dtype=np.complex128
    )
    rhs = (
        z_train.conj().T @ y_train
        + reg_normal_eq * reference_readout.T
    )
    try:
        fitted_readout_t = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        fitted_readout_t = np.linalg.pinv(gram) @ rhs
    return z_test @ fitted_readout_t


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
    wesn_lite_energy: float = 0.80,
    wesn_lite_readout_reg: float = 1e-4,
    wesn_lite_readout_mode: str = "centered-ridge",
    input_scale: float = 0.5,
    window_length: int = 2,
    subcarriers_per_rb: int = 12,
    process_snr_db: float | None = None,
) -> tuple[float, float, float, float, float, dict]:
    # Balanced WESN-Lite uses production matched-ridge readout. Keep the
    # legacy CLI knobs so existing command lines still parse.
    _ = (wesn_lite_readout_reg, wesn_lite_readout_mode)
    rng = np.random.default_rng(seed)

    # State equation: x_t already comes from ns-3. Extra temporally white
    # process noise w_t is added to that true CSI when process_snr_db is set:
    #   x_t <- x_t + w_t,  Q = E[|x|^2] / 10^(process_snr_db/10)
    # Observation equation is unchanged: y_t = x_t + v_t with variance from snr_db.
    h_true_dec = h_clean_dec
    if process_snr_db is not None and np.isfinite(float(process_snr_db)):
        h_true_dec, _q_var = add_complex_awgn(h_clean_dec, float(process_snr_db), rng)
    h_noisy_dec, noise_var = add_complex_awgn(h_true_dec, snr_db, rng)
    hist_clean = channel_seq_to_history(h_true_dec)
    hist_noisy = channel_seq_to_history(h_noisy_dec)
    err_hist = np.full(hist_noisy.shape, noise_var, dtype=np.float32)

    t_dec = hist_clean.shape[0]
    _, num_rx, num_tx, _, num_sc = h_clean_dec.shape
    online_history_len = int(history_len)
    min_config_len = max(int(window_length) + 1, 3)
    if online_history_len < max(int(window_length) + 1, 2):
        raise ValueError(
            f"history_len={online_history_len} must be at least window_length+1="
            f"{int(window_length) + 1}."
        )
    if t_dec <= min_config_len + 1:
        raise ValueError("Not enough decimated samples for an offline/online split.")
    if not (0.0 < float(offline_ratio) <= 1.0):
        raise ValueError(f"offline_ratio must be in (0, 1], got {offline_ratio}.")

    if float(offline_ratio) >= 1.0:
        offline_len = t_dec
        offline_h = hist_noisy
        offline_e = err_hist
        first_target = online_history_len
    else:
        offline_len = int(np.floor(t_dec * float(offline_ratio)))
        offline_len = max(1, min(offline_len, t_dec - 1))
        online_len = t_dec - offline_len
        if offline_len < min_config_len:
            raise ValueError(
                f"Offline phase too short for statistics collection: offline_len={offline_len}, "
                f"need at least {min_config_len}."
            )
        if online_len < 2:
            raise ValueError(
                f"Online phase too short for prediction evaluation: online_len={online_len}."
            )
        offline_h = hist_noisy[:offline_len]
        offline_e = err_hist[:offline_len]
        # Online CSI windows use rc_config.history_len. The offline segment is
        # only used to configure W_in/W_res (WESN) or the initial KF models.
        first_target = offline_len

    scored_targets = range(first_target + 1, t_dec)
    if first_target < online_history_len:
        raise ValueError(
            f"Need history_len={online_history_len} samples before the first online "
            f"target at index {first_target}. Reduce --history-len or load a longer "
            "slot range."
        )
    if not scored_targets:
        raise ValueError("No online targets remain after the production warmup slot.")

    wesn_kwargs = dict(
        esn_m=num_basis,
        esn_k=rp_degree,
        num_freqs=num_freqs,
        activation=activation,
        ls_reg=ls_reg,
        hankel_energy=wesn_lite_energy,
        input_scale=input_scale,
        window_length=window_length,
        subcarriers_per_rb=subcarriers_per_rb,
    )

    def _build_wesn(*, lite: bool, random: bool) -> configured_wesn_pred:
        predictor = configured_wesn_pred(
            make_wesn_rc_config(**wesn_kwargs, lite=lite, random=random),
            num_sc,
            num_rx,
            num_tx,
        )
        predictor.fit_offline(offline_h, offline_e)
        _freeze_predictor_weights(predictor)
        return predictor

    balanced_wesn = _build_wesn(lite=False, random=False)
    balanced_wesn_lite = _build_wesn(lite=True, random=False)
    random_wesn = _build_wesn(lite=False, random=True)

    kalman = kalman_filter_pred(
        ar_order=ar_order,
        debug=False,
        reconfiguration_interval=1,
    )
    kalman.num_bs_ant = int(num_rx)
    kalman.num_ue_ant = int(max(num_tx, 1))
    # Production initializes F/Q/R from the drop-local offline segment and
    # keeps that predictor for the online loop. reconfiguration_interval=1
    # still refits from each current causal window.
    kalman.predict(offline_h, offline_e)

    # Production SS-KF: F, Q, R, and the Riccati gain are estimated once from
    # the offline half and then frozen. Online predict() only applies that
    # fixed gain; it does not re-estimate the model.
    ss_kalman = steady_state_kalman_filter_pred(ar_order=ar_order)
    ss_kalman.fit_offline(offline_h, offline_e)

    wesn_lite_rank_summary = balanced_wesn_rank_summary(balanced_wesn_lite)
    diag_info = (
        {
            "balanced_wesn_metrics": dict(balanced_wesn.predictor_complexity_metrics),
            "balanced_wesn_lite_metrics": dict(
                balanced_wesn_lite.predictor_complexity_metrics
            ),
            "wesn_lite_rank_summary": wesn_lite_rank_summary,
        }
        if run_diagnostics
        else None
    )

    num_full = 0.0
    den_full = 0.0
    num_ss = 0.0
    den_ss = 0.0
    num_cfg = 0.0
    den_cfg = 0.0
    num_lite = 0.0
    den_lite = 0.0
    num_rand = 0.0
    den_rand = 0.0

    def _causal_history(target_idx: int) -> tuple[np.ndarray, np.ndarray]:
        start = target_idx - online_history_len
        return hist_noisy[start:target_idx], err_hist[start:target_idx]

    # Production still runs the first online cycle (warmup / start_slot_idx)
    # but excludes it from averaged NMSE.
    y_warm, e_warm = _causal_history(first_target)
    kalman.predict(y_warm, e_warm)
    ss_kalman.predict(y_warm)
    balanced_wesn.predict(y_warm, e_warm)
    balanced_wesn_lite.predict(y_warm, e_warm)
    random_wesn.predict(y_warm, e_warm)

    for target_idx in scored_targets:
        y_hist, e_hist_win = _causal_history(target_idx)
        x_true = hist_clean[target_idx]

        x_pred_full = kalman.predict(y_hist, e_hist_win)
        x_pred_ss = ss_kalman.predict(y_hist)
        x_pred_cfg = balanced_wesn.predict(y_hist, e_hist_win)
        x_pred_lite = balanced_wesn_lite.predict(y_hist, e_hist_win)
        x_pred_rand = random_wesn.predict(y_hist, e_hist_win)

        num_full += float(np.sum(np.abs(x_pred_full - x_true) ** 2))
        den_full += float(np.sum(np.abs(x_true) ** 2))
        num_ss += float(np.sum(np.abs(x_pred_ss - x_true) ** 2))
        den_ss += float(np.sum(np.abs(x_true) ** 2))
        num_cfg += float(np.sum(np.abs(x_pred_cfg - x_true) ** 2))
        den_cfg += float(np.sum(np.abs(x_true) ** 2))
        num_lite += float(np.sum(np.abs(x_pred_lite - x_true) ** 2))
        den_lite += float(np.sum(np.abs(x_true) ** 2))
        num_rand += float(np.sum(np.abs(x_pred_rand - x_true) ** 2))
        den_rand += float(np.sum(np.abs(x_true) ** 2))

    nmse_full = num_full / max(den_full, 1e-15)
    nmse_ss = num_ss / max(den_ss, 1e-15)
    nmse_cfg = num_cfg / max(den_cfg, 1e-15)
    nmse_lite = num_lite / max(den_lite, 1e-15)
    nmse_rand = num_rand / max(den_rand, 1e-15)

    if run_diagnostics and diag_info is not None:
        print("\n[ESN diagnostics]")
        cfg_metrics = diag_info["balanced_wesn_metrics"]
        lite_metrics = diag_info["balanced_wesn_lite_metrics"]
        print(
            "Balanced Configured WESN: "
            f"method={cfg_metrics.get('method')}, "
            f"state_dim={cfg_metrics.get('state_dimension')}, "
            f"feature_dim={cfg_metrics.get('feature_dimension')}, "
            f"orders={cfg_metrics.get('balanced_orders_per_mode')}, "
            f"RBs={cfg_metrics.get('model_num_resource_blocks')}"
        )
        lite_rank = diag_info["wesn_lite_rank_summary"]
        print(
            "Balanced Configured WESN-Lite: "
            f"method={lite_metrics.get('method')}, "
            f"state_dim={lite_rank['state_dimension']}, "
            f"Hankel order mean={lite_rank['mean']:.2f}, "
            f"mode={lite_rank['mode']}, range=[{lite_rank['min']}, {lite_rank['max']}], "
            f"retained Hankel energy mean={lite_rank['retained_energy_mean']:.3f}"
        )
        print("[/ESN diagnostics]\n")

    return (
        nmse_ss,
        nmse_full,
        nmse_cfg,
        nmse_lite,
        nmse_rand,
        wesn_lite_rank_summary,
    )



def main():
    parser = argparse.ArgumentParser(description="P2P Kalman channel prediction NMSE vs SNR on ns3 saved channels")
    parser.add_argument("--ns3-root", type=str, default="ns3", help="Root folder containing channels_<mobility>_<drop>")
    parser.add_argument("--mobility", type=str, default="higher_mobility", help="Mobility folder key")
    parser.add_argument("--drop-idx", type=int, default=1)
    parser.add_argument("--start-slot", type=int, default=1)
    parser.add_argument("--end-slot", type=int, default=100)
    parser.add_argument("--feedback-delay", type=int, default=4)
    parser.add_argument("--ar-order", type=int, default=2)
    parser.add_argument("--esn-m", "--fb-m", dest="esn_m", type=int, default=2, help="PCA modes for balanced WESN (production esn_m=2)")
    parser.add_argument("--esn-k", "--fb-k", dest="esn_k", type=int, default=4, help="Poles per mode / balanced order factor (production esn_k=4)")
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
        default=1e-4,
        help="Ridge regularization used by ESN readout solve",
    )
    parser.add_argument(
        "--wesn-lite-energy",
        type=float,
        default=0.80,
        help="Hankel-energy threshold used by Balanced Configured WESN-Lite (production BALANCED_LITE_HANKEL_ENERGY).",
    )
    parser.add_argument(
        "--wesn-lite-readout-reg",
        type=float,
        default=1e-4,
        help="Unused by balanced WESN-Lite; kept for command-line compatibility.",
    )
    parser.add_argument(
        "--wesn-lite-readout-mode",
        choices=["matched-ridge", "centered-ridge"],
        default="centered-ridge",
        help="Unused by balanced WESN-Lite; kept for command-line compatibility.",
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=0.5,
        help="WESN input scale (production input_scale=0.5).",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=8,
        help="Online CSI history length fed to WESN and Kalman (rc_config.history_len).",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=2,
        help="WESN/Kalman window length (production window_length=2).",
    )
    parser.add_argument(
        "--subcarriers-per-rb",
        type=int,
        default=12,
        help=(
            "Resource-block averaging width. Matches "
            "WESN_LITE_SUBCARRIERS_PER_RB=12 in twc_tput_across_tx.sh."
        ),
    )
    parser.add_argument("--rx-ant", type=int, default=4)
    parser.add_argument("--tx-ant", type=int, default=4)
    parser.add_argument("--snr-start", type=int, default=0)
    parser.add_argument("--snr-stop", type=int, default=26)
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
        help="Print balanced-WESN configuration diagnostics.",
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
    t_dec = int(h_clean_dec.shape[0])
    if float(args.offline_ratio) >= 1.0:
        offline_len = t_dec
    else:
        offline_len = int(np.floor(t_dec * float(args.offline_ratio)))
        offline_len = max(1, min(offline_len, t_dec - 1))
    print(
        "Within-drop protocol: "
        f"{offline_len} offline / {t_dec - offline_len} online slots; "
        f"online CSI history length={args.history_len}; "
        f"RB width={args.subcarriers_per_rb}, activation={args.esn_activation}, "
        f"m={args.esn_m}, k={args.esn_k}, input_scale={args.input_scale}."
    )

    snr_vals = np.arange(args.snr_start, args.snr_stop + 1, args.snr_step)
    nmse_ss_vals = []
    nmse_full_vals = []
    nmse_cfg_vals = []
    nmse_lite_vals = []
    nmse_rand_vals = []
    wesn_lite_rank_mean_vals = []
    wesn_lite_rank_mode_vals = []
    wesn_lite_state_dim_vals = []

    for snr_db in snr_vals:
        (
            nmse_ss,
            nmse_full,
            nmse_cfg,
            nmse_lite,
            nmse_rand,
            lite_rank,
        ) = evaluate_nmse_over_chunks(
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
            wesn_lite_energy=float(args.wesn_lite_energy),
            wesn_lite_readout_reg=float(args.wesn_lite_readout_reg),
            wesn_lite_readout_mode=str(args.wesn_lite_readout_mode),
            input_scale=float(args.input_scale),
            window_length=int(args.window_length),
            subcarriers_per_rb=int(args.subcarriers_per_rb),
        )
        nmse_ss_vals.append(nmse_ss)
        nmse_full_vals.append(nmse_full)
        nmse_cfg_vals.append(nmse_cfg)
        nmse_lite_vals.append(nmse_lite)
        nmse_rand_vals.append(nmse_rand)
        wesn_lite_rank_mean_vals.append(lite_rank["mean"])
        wesn_lite_rank_mode_vals.append(lite_rank["mode"])
        wesn_lite_state_dim_vals.append(lite_rank["state_dimension"])
        print(
            f"SNR={snr_db:>2d} dB | NMSE steady_state_kf={nmse_ss:.4e}, full={nmse_full:.4e}, "
            f"configured_wesn_balanced={nmse_cfg:.4e}, "
            f"configured_wesn_balanced_lite={nmse_lite:.4e}, "
            f"random_esn={nmse_rand:.4e} | Balanced WESN-Lite "
            f"Hankel order mean={lite_rank['mean']:.2f}, mode={lite_rank['mode']}, "
            f"R={lite_rank['state_dimension']}"
        )

        hold = 1

    nmse_ss_vals = np.asarray(nmse_ss_vals)
    nmse_full_vals = np.asarray(nmse_full_vals)
    nmse_cfg_vals = np.asarray(nmse_cfg_vals)
    nmse_lite_vals = np.asarray(nmse_lite_vals)
    nmse_rand_vals = np.asarray(nmse_rand_vals)

    out_path = Path(args.plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_path.with_name(
        out_path.stem + f"_activation_{args.esn_activation}_{args.mobility}" + out_path.suffix
    )

    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    ax.plot(
        snr_vals,
        nmse_full_vals,
        marker="s",
        color="tab:orange",
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Kalman Filter",
    )
    ax.plot(
        snr_vals,
        nmse_ss_vals,
        marker="P",
        color="tab:purple",
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Steady-State KF",
    )
    ax.plot(
        snr_vals,
        nmse_cfg_vals,
        marker="^",
        color="tab:green",
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Balanced Configured WESN",
    )
    ax.plot(
        snr_vals,
        nmse_lite_vals,
        marker="v",
        color="tab:brown",
        linestyle="--",
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Balanced Configured WESN-Lite",
    )
    ax.plot(
        snr_vals,
        nmse_rand_vals,
        marker="d",
        color="0.45",
        linewidth=2.0,
        markersize=6.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Random WESN",
    )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Channel prediction NMSE")

    ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, length=5)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5)

    ax.legend(frameon=False, loc="upper right", handlelength=1.8)
    fig.tight_layout(pad=0.2)

    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved plot to: {out_path}")

    np.savez(
        save_path,
        snr_vals=snr_vals,
        nmse_ss_vals=nmse_ss_vals,
        nmse_full_vals=nmse_full_vals,
        nmse_cfg_vals=nmse_cfg_vals,
        nmse_lite_vals=nmse_lite_vals,
        nmse_rand_vals=nmse_rand_vals,
        wesn_lite_energy=np.asarray(args.wesn_lite_energy),
        wesn_lite_readout_reg=np.asarray(args.wesn_lite_readout_reg),
        wesn_lite_readout_mode=np.asarray(args.wesn_lite_readout_mode),
        wesn_lite_rank_mean_vals=np.asarray(wesn_lite_rank_mean_vals),
        wesn_lite_rank_mode_vals=np.asarray(wesn_lite_rank_mode_vals),
        wesn_lite_state_dim_vals=np.asarray(wesn_lite_state_dim_vals),
    )
    print(f"Saved raw NMSE data to: {save_path}")


if __name__ == "__main__":
    main()
