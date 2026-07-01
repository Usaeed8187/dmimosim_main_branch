#!/usr/bin/env python3
"""Low-complexity optimized ZF under per-antenna power constraints.

This standalone script generates a time-evolving distributed-MIMO downlink channel
with Sionna TR 38.901 models. Each UE is still a multi-antenna UE, but for the
ZF design we reduce each UE's aggregate MIMO channel to its dominant scalar
effective channel:

    g_k^H = sigma_k v_k^H

where v_k is the first right singular vector of H_k and sigma_k is the dominant
singular value. The script compares:

  1) Naive ZF using G = [g_1^H; ...; g_K^H], followed by one common power backoff.
  2) A low-complexity regularized-dual optimized ZF method inspired by:
     Li, Dam, Teo, Cantoni, "A Low Complexity Optimization Algorithm for
     Zero-Forcing Precoding Under Per-Antenna Power Constraints", ICASSP 2015.

The paper is written for per-antenna constraints. Here we use the paper's
per-antenna PAPC structure directly. A D-MIMO transmitter's total power is
split equally across its local antennas, e.g., with tx-powers 1.0,0.5,0.25
and antennas-per-tx=4, the per-antenna limits are
1.0/4, 0.5/4, and 0.25/4 for the three transmitters.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Config:
    num_drops: int = 20
    num_slots: int = 2000
    num_tx: int = 3
    antennas_per_tx: int = 4
    num_users: int = 4
    rx_antennas_per_user: int = 2
    snr_db: float = 10.0
    seed: int = 7

    # Sionna / TR 38.901 channel controls
    sionna_scenario: str = "umi"  # umi, uma, rma
    carrier_frequency_hz: float = 3.5e9
    subcarrier_spacing_hz: float = 30e3
    slot_duration_s: float = 1e-3
    ue_speed_kmh: float = 30.0
    bs_height_m: float = 25.0
    ue_height_m: float = 1.5
    cell_radius_m: float = 100.0
    min_ue_distance_m: float = 10.0
    tx_spacing_m: float = 50.0
    sionna_o2i_model: str = "low"
    sionna_enable_pathloss: bool = True
    sionna_enable_shadow_fading: bool = False
    normalize_average_channel_power: bool = True

    moving_avg_window: int = 100

    # Low-complexity PAPC-ZF solver controls.
    # This version uses per-antenna PAPC, matching the paper's structure.
    # If papc_rho <= 0, rho is chosen from the paper's beta/error-bound rule.
    # Dual variables are updated using equation-(22)-style Lipschitz steps.
    papc_num_iters: int = 1000
    papc_rho: float = -1.0
    papc_beta: float = 0.3
    papc_tol: float = 1e-5
    papc_lipschitz_floor: float = 1e-12
    papc_final_safety_backoff: bool = False

    output_dir: Path = Path("results_low_complexity_papc_zf")


def parse_tx_powers(s: str, num_tx: int) -> np.ndarray:
    vals = np.array([float(x.strip()) for x in s.split(",") if x.strip()], dtype=np.float64)
    if vals.size == 1:
        vals = np.repeat(vals[0], num_tx)
    if vals.size != num_tx:
        raise ValueError(f"Expected one power or {num_tx} comma-separated powers, got {vals.size}.")
    if np.any(vals <= 0):
        raise ValueError("All transmitter powers must be positive.")
    return vals



def per_antenna_powers_from_tx_powers(tx_powers: np.ndarray, antennas_per_tx: int) -> np.ndarray:
    """Split each distributed transmitter's total power equally over its antennas."""
    tx_powers = np.asarray(tx_powers, dtype=np.float64)
    if antennas_per_tx <= 0:
        raise ValueError("antennas_per_tx must be positive.")
    return np.repeat(tx_powers / float(antennas_per_tx), int(antennas_per_tx))


def per_tx_block_powers_from_precoder(
    precoder: np.ndarray,
    num_tx: int,
    antennas_per_tx: int,
) -> np.ndarray:
    """Compute total used power per distributed transmitter block."""
    powers = np.zeros(num_tx, dtype=np.float64)
    for b in range(num_tx):
        block = precoder[b * antennas_per_tx : (b + 1) * antennas_per_tx, :]
        powers[b] = float(np.real(np.sum(np.abs(block) ** 2)))
    return powers


def per_antenna_powers_from_precoder(precoder: np.ndarray) -> np.ndarray:
    """Compute total stream power used by every transmit antenna."""
    return np.real(np.sum(np.abs(precoder) ** 2, axis=1)).astype(np.float64, copy=False)


def common_per_antenna_backoff(
    p_raw: np.ndarray,
    per_antenna_powers: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Optional safety scaling for per-antenna constraints."""
    used = per_antenna_powers_from_precoder(p_raw)
    alpha = float(np.min(np.sqrt(per_antenna_powers / np.maximum(used, eps))))
    alpha = min(alpha, 1.0)
    return alpha * p_raw, alpha, used



def _rough_rectangular_panel_dims(num_antennas: int) -> tuple[int, int]:
    n = max(1, int(num_antennas))
    rows = int(np.floor(np.sqrt(n)))
    rows = max(1, rows)
    cols = int(np.ceil(n / rows))
    return rows, cols


def _make_sionna_panel_array(num_antennas: int, carrier_frequency_hz: float, is_bs: bool):
    """Build a simple single-polarized Sionna PanelArray.

    For non-rectangular antenna counts, Sionna creates the nearest larger panel;
    the generated channel is sliced back down to the requested count afterward.
    """
    try:
        from sionna.channel.tr38901 import PanelArray
    except ImportError:  # Sionna >= 1/2 namespace
        from sionna.phy.channel.tr38901 import PanelArray

    rows, cols = _rough_rectangular_panel_dims(num_antennas)
    pattern = "38.901" if is_bs else "omni"
    return PanelArray(
        num_rows_per_panel=rows,
        num_cols_per_panel=cols,
        polarization="single",
        polarization_type="V",
        antenna_pattern=pattern,
        carrier_frequency=carrier_frequency_hz,
    )


def _sample_dmimo_sionna_topology(cfg: Config, rng: np.random.Generator):
    """Sample B distributed BS/TX locations and K UE locations/velocities."""
    tx_x = (np.arange(cfg.num_tx) - (cfg.num_tx - 1) / 2.0) * float(cfg.tx_spacing_m)
    bs_loc = np.zeros((1, cfg.num_tx, 3), dtype=np.float32)
    bs_loc[0, :, 0] = tx_x
    bs_loc[0, :, 1] = 0.0
    bs_loc[0, :, 2] = float(cfg.bs_height_m)

    min_r = max(1.0, float(cfg.min_ue_distance_m))
    max_r = max(min_r + 1.0, float(cfg.cell_radius_m))
    radii = np.sqrt(rng.uniform(min_r**2, max_r**2, size=cfg.num_users))
    angles = rng.uniform(-np.pi, np.pi, size=cfg.num_users)
    ut_loc = np.zeros((1, cfg.num_users, 3), dtype=np.float32)
    ut_loc[0, :, 0] = radii * np.cos(angles)
    ut_loc[0, :, 1] = radii * np.sin(angles)
    ut_loc[0, :, 2] = float(cfg.ue_height_m)

    ut_orientations = np.zeros((1, cfg.num_users, 3), dtype=np.float32)
    bs_orientations = np.zeros((1, cfg.num_tx, 3), dtype=np.float32)

    speed_mps = float(cfg.ue_speed_kmh) / 3.6
    velocity_angles = rng.uniform(-np.pi, np.pi, size=cfg.num_users)
    ut_velocities = np.zeros((1, cfg.num_users, 3), dtype=np.float32)
    ut_velocities[0, :, 0] = speed_mps * np.cos(velocity_angles)
    ut_velocities[0, :, 1] = speed_mps * np.sin(velocity_angles)

    # Keep all UEs outdoor for this small comparison script.
    in_state = np.zeros((1, cfg.num_users), dtype=bool)
    return ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state


def simulate_sionna_dmimo_channels(cfg: Config, rng: np.random.Generator, drop_idx: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate flat per-slot D-MIMO channels from Sionna TR 38.901.

    Returns
    -------
    channels:
        Shape [T, K, Nrx, B, Ntb]. The aggregate channel for UE k is obtained
        by horizontally concatenating channels[t,k,:,b,:] over b.
    bs_loc:
        Shape [B, 3].
    ut_loc:
        Shape [K, 3].
    """
    try:
        import tensorflow as tf
        try:
            from sionna.channel.tr38901 import UMi, UMa, RMa
            from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
        except ImportError:  # Sionna >= 1/2 namespace
            from sionna.phy.channel.tr38901 import UMi, UMa, RMa
            from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
    except ImportError as exc:
        raise ImportError(
            "This script is Sionna-only. Install Sionna/TensorFlow in this environment "
            "or run it inside the environment where your existing Sionna scripts work."
        ) from exc

    tf.random.set_seed(int(cfg.seed) + 1000003 * int(drop_idx))
    ut_array = _make_sionna_panel_array(
        cfg.rx_antennas_per_user,
        cfg.carrier_frequency_hz,
        is_bs=False,
    )
    bs_array = _make_sionna_panel_array(
        cfg.antennas_per_tx,
        cfg.carrier_frequency_hz,
        is_bs=True,
    )

    model_name = cfg.sionna_scenario.lower()
    model_cls = {"umi": UMi, "uma": UMa, "rma": RMa}.get(model_name)
    if model_cls is None:
        raise ValueError("--sionna-scenario must be one of: umi, uma, rma.")

    channel_model = model_cls(
        carrier_frequency=float(cfg.carrier_frequency_hz),
        o2i_model=str(cfg.sionna_o2i_model),
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        enable_pathloss=bool(cfg.sionna_enable_pathloss),
        enable_shadow_fading=bool(cfg.sionna_enable_shadow_fading),
    )

    topo = _sample_dmimo_sionna_topology(cfg, rng)
    ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topo
    try:
        channel_model.set_topology(
            ut_loc=tf.constant(ut_loc),
            bs_loc=tf.constant(bs_loc),
            ut_orientations=tf.constant(ut_orientations),
            bs_orientations=tf.constant(bs_orientations),
            ut_velocities=tf.constant(ut_velocities),
            in_state=tf.constant(in_state),
        )
    except TypeError:
        # Older Sionna releases used a more positional-friendly signature.
        channel_model.set_topology(
            tf.constant(ut_loc),
            tf.constant(bs_loc),
            tf.constant(ut_orientations),
            tf.constant(bs_orientations),
            tf.constant(ut_velocities),
            tf.constant(in_state),
        )

    sampling_frequency = 1.0 / max(float(cfg.slot_duration_s), 1e-12)
    print(
        f"Generating Sionna {model_name.upper()} D-MIMO channels for drop {drop_idx + 1}: "
        f"B={cfg.num_tx}, Ntb={cfg.antennas_per_tx}, K={cfg.num_users}, "
        f"Nrx={cfg.rx_antennas_per_user}, slots/drop={cfg.num_slots}, "
        f"fc={cfg.carrier_frequency_hz/1e9:.3f} GHz, speed={cfg.ue_speed_kmh:.2f} km/h"
    )
    a, tau = channel_model(
        num_time_samples=int(cfg.num_slots),
        sampling_frequency=float(sampling_frequency),
    )

    # Use a tiny 3-subcarrier grid and select the center RE, matching your
    # existing narrowband toy-test interface.
    freqs = subcarrier_frequencies(3, float(cfg.subcarrier_spacing_hz))
    h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=False)
    h_np = np.asarray(h_freq.numpy())

    # Expected Sionna shape:
    # [batch, num_ut, rx_ant, num_bs, tx_ant, num_time_samples, num_subcarriers]
    if h_np.ndim != 7:
        raise RuntimeError(f"Unexpected Sionna OFDM channel rank {h_np.ndim}; shape={h_np.shape}")

    h_center = h_np[
        0,
        : cfg.num_users,
        : cfg.rx_antennas_per_user,
        : cfg.num_tx,
        : cfg.antennas_per_tx,
        : cfg.num_slots,
        1,
    ]
    channels = np.transpose(h_center, (4, 0, 1, 2, 3)).astype(np.complex128, copy=False)
    expected = (
        cfg.num_slots,
        cfg.num_users,
        cfg.rx_antennas_per_user,
        cfg.num_tx,
        cfg.antennas_per_tx,
    )
    if channels.shape != expected:
        raise RuntimeError(f"Unexpected generated channel shape {channels.shape}; expected {expected}.")

    if cfg.normalize_average_channel_power:
        rms = float(np.sqrt(np.mean(np.abs(channels) ** 2)))
        channels = channels / max(rms, 1e-12)

    return channels, bs_loc[0].astype(np.float64), ut_loc[0].astype(np.float64)


def aggregate_channel(user_link_channels: np.ndarray) -> np.ndarray:
    """Convert one UE channel [Nrx, B, Ntb] to aggregate [Nrx, B*Ntb]."""
    nrx, num_tx, ntb = user_link_channels.shape
    return user_link_channels.reshape(nrx, num_tx * ntb)


def aggregate_all_users(slot_channels: np.ndarray) -> np.ndarray:
    """Convert slot channels [K, Nrx, B, Ntb] to [K, Nrx, Ntot]."""
    return np.stack([aggregate_channel(slot_channels[k]) for k in range(slot_channels.shape[0])], axis=0)



def dominant_effective_channel_rows(h_agg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one effective MISO row per multi-antenna UE.

    For UE k:
        H_k = U_k Sigma_k V_k^H
        g_k^H = sigma_k v_k^H

    Returns
    -------
    g_rows:
        Shape [K, Ntot]. Row k is g_k^H = sigma_k v_k^H.
    v_right:
        Shape [K, Ntot]. Row k stores v_k, the dominant right singular vector.
    sigma:
        Shape [K]. Dominant singular value for each UE.
    """
    g_rows = []
    v_right = []
    sigma = []
    for k in range(h_agg.shape[0]):
        _u, s, vh = np.linalg.svd(h_agg[k], full_matrices=True)
        vk = vh.conj().T[:, 0]
        vk = vk / max(np.linalg.norm(vk), 1e-12)
        sk = float(s[0]) if s.size else 0.0
        v_right.append(vk)
        sigma.append(sk)
        g_rows.append(sk * vk.conj())
    return (
        np.stack(g_rows, axis=0).astype(np.complex128, copy=False),
        np.stack(v_right, axis=0).astype(np.complex128, copy=False),
        np.asarray(sigma, dtype=np.float64),
    )


def common_power_backoff(
    p_raw: np.ndarray,
    tx_powers: np.ndarray,
    antennas_per_tx: int,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Scale the whole distributed precoder by one scalar to meet every TX budget."""
    num_tx = tx_powers.size
    block_powers = np.zeros(num_tx, dtype=np.float64)
    for b in range(num_tx):
        block = p_raw[b * antennas_per_tx : (b + 1) * antennas_per_tx, :]
        block_powers[b] = float(np.real(np.sum(np.abs(block) ** 2)))
    alpha = float(np.min(np.sqrt(tx_powers / np.maximum(block_powers, eps))))
    alpha = min(alpha, 1.0)  # do not boost a low-power raw solution
    return alpha * p_raw, alpha, block_powers


def build_naive_zf_precoder(
    g_rows: np.ndarray,
    tx_powers: np.ndarray,
    antennas_per_tx: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Ordinary pseudoinverse ZF on effective rows g_k^H, then common power backoff.

    If G is [g_1^H; ...; g_K^H], this makes G P ~= I before power backoff.
    """
    p_raw = np.linalg.pinv(g_rows)  # [Ntot, K]
    return common_power_backoff(p_raw, tx_powers, antennas_per_tx)


def _complex_row_real_coeffs(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return real-vector coefficients for Re{row @ w} and Im{row @ w}.

    For row = a + j b and w = x + j y:
        Re{row @ w} = a^T x - b^T y
        Im{row @ w} = b^T x + a^T y
    """
    a = np.real(row)
    b = np.imag(row)
    return np.concatenate([a, -b]), np.concatenate([b, a])


def _build_real_constraint_matrices(g_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build H1 and H2 for the real-valued optimized-ZF formulation.

    Variables are stacked as:
        x = [Re w_0, Im w_0, Re w_1, Im w_1, ..., Re w_{K-1}, Im w_{K-1}].

    H1 x + t 1 <= 0 encodes Re{g_k^H w_k} >= t.
    H2 x = 0 encodes Im{g_k^H w_k}=0 and g_j^H w_k=0 for j != k.
    """
    num_users, ntot = g_rows.shape
    nvar = 2 * ntot * num_users

    h1_rows = []
    h2_rows = []

    for k in range(num_users):
        re_coeff, im_coeff = _complex_row_real_coeffs(g_rows[k])

        row = np.zeros(nvar, dtype=np.float64)
        off = k * 2 * ntot
        row[off : off + 2 * ntot] = -re_coeff
        h1_rows.append(row)

        row = np.zeros(nvar, dtype=np.float64)
        row[off : off + 2 * ntot] = im_coeff
        h2_rows.append(row)

    for tx_user in range(num_users):
        off = tx_user * 2 * ntot
        for rx_user in range(num_users):
            if rx_user == tx_user:
                continue
            re_coeff, im_coeff = _complex_row_real_coeffs(g_rows[rx_user])

            row_re = np.zeros(nvar, dtype=np.float64)
            row_re[off : off + 2 * ntot] = re_coeff
            h2_rows.append(row_re)

            row_im = np.zeros(nvar, dtype=np.float64)
            row_im[off : off + 2 * ntot] = im_coeff
            h2_rows.append(row_im)

    return np.vstack(h1_rows), np.vstack(h2_rows)


def _x_to_precoder(x: np.ndarray, ntot: int, num_users: int) -> np.ndarray:
    """Convert real stacked vector x back to complex precoder W [Ntot, K]."""
    w = np.zeros((ntot, num_users), dtype=np.complex128)
    for k in range(num_users):
        off = k * 2 * ntot
        re = x[off : off + ntot]
        im = x[off + ntot : off + 2 * ntot]
        w[:, k] = re + 1j * im
    return w


def _per_antenna_powers_from_x(
    x: np.ndarray,
    num_users: int,
    ntot: int,
) -> np.ndarray:
    """Compute per-antenna powers from real stacked x."""
    powers = np.zeros(ntot, dtype=np.float64)
    for k in range(num_users):
        off = k * 2 * ntot
        re = x[off : off + ntot]
        im = x[off + ntot : off + 2 * ntot]
        powers += re**2 + im**2
    return powers


def _tx_block_powers_from_x(
    x: np.ndarray,
    num_users: int,
    num_tx: int,
    antennas_per_tx: int,
) -> np.ndarray:
    """Compute per-TX block powers from real stacked x."""
    ntot = num_tx * antennas_per_tx
    ant_powers = _per_antenna_powers_from_x(x, num_users, ntot)
    return ant_powers.reshape(num_tx, antennas_per_tx).sum(axis=1)


def _denom_from_mu(
    mu: np.ndarray,
    num_users: int,
    antennas_per_tx: int,
    rho: float,
) -> np.ndarray:
    """Diagonal of rho I + sum_n mu_n A_n for real stacked variables.

    mu is per-antenna here, matching the paper. It is repeated for real/imag
    variables and then repeated for each user stream.
    """
    one_user = np.concatenate([mu, mu])
    return float(rho) + np.tile(one_user, num_users)


def _regularized_dual_primal_minimizer(
    h1: np.ndarray,
    h2: np.ndarray,
    lam: np.ndarray,
    v: np.ndarray,
    mu: np.ndarray,
    num_users: int,
    antennas_per_tx: int,
    rho: float,
) -> tuple[float, np.ndarray]:
    """Closed-form minimizer of the regularized Lagrangian over (t, x).

    This is the direct analogue of equations (14)-(15) in the paper, with
    per-antenna constraints replacing per-antenna constraints.
    """
    t_val = (1.0 - float(np.sum(lam))) / (2.0 * rho)
    c = h1.T @ lam + h2.T @ v
    denom = _denom_from_mu(mu, num_users, antennas_per_tx, rho)
    x = -0.5 * c / np.maximum(denom, 1e-12)
    return t_val, x


def _regularized_dual_value_and_gradient(
    h1: np.ndarray,
    h2: np.ndarray,
    lam: np.ndarray,
    v: np.ndarray,
    mu: np.ndarray,
    per_antenna_powers: np.ndarray,
    num_users: int,
    antennas_per_tx: int,
    rho: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Evaluate regularized dual d_rho and its gradients.

    This follows the paper's per-antenna PAPC form:
        grad_mu[n] = x^T A_n x - P_n.
    """
    ntot = per_antenna_powers.size
    t_val, x = _regularized_dual_primal_minimizer(
        h1, h2, lam, v, mu, num_users, antennas_per_tx, rho
    )
    grad_lam = h1 @ x + t_val
    grad_v = h2 @ x
    antenna_powers = _per_antenna_powers_from_x(x, num_users, ntot)
    grad_mu = antenna_powers - per_antenna_powers

    # Direct L_rho evaluation at the minimizing (t, x).
    d_val = (
        -t_val
        + float(lam @ grad_lam)
        + float(v @ grad_v)
        + float(mu @ grad_mu)
        + rho * float(t_val * t_val + x @ x)
    )
    return d_val, grad_lam, grad_v, grad_mu, t_val, x, antenna_powers


def _project_dual(
    lam: np.ndarray,
    v: np.ndarray,
    mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projection onto lambda >= 0, mu >= 0; v is unconstrained."""
    return np.maximum(lam, 0.0), v, np.maximum(mu, 0.0)


def _dual_projected_gradient_step(
    lam: np.ndarray,
    v: np.ndarray,
    mu: np.ndarray,
    grad_lam: np.ndarray,
    grad_v: np.ndarray,
    grad_mu: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One projected gradient-ascent step for the regularized dual problem."""
    return _project_dual(
        lam + step * grad_lam,
        v + step * grad_v,
        mu + step * grad_mu,
    )



def _dual_projected_lipschitz_step(
    lam: np.ndarray,
    v: np.ndarray,
    mu: np.ndarray,
    grad_lam: np.ndarray,
    grad_v: np.ndarray,
    grad_mu: np.ndarray,
    l_lam: float,
    l_v: float,
    l_mu: float,
    floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equation-(22)-style projected dual update with Lipschitz steps.

    lambda <- [lambda + grad_lambda/L_lambda]_+
    v      <-  v      + grad_v/L_v
    mu     <- [mu     + grad_mu/L_mu]_+
    """
    l_lam = max(float(l_lam), float(floor))
    l_v = max(float(l_v), float(floor))
    l_mu = max(float(l_mu), float(floor))
    return _project_dual(
        lam + grad_lam / l_lam,
        v + grad_v / l_v,
        mu + grad_mu / l_mu,
    )


def _max_eigval_symmetric(mat: np.ndarray, floor: float = 1e-12) -> float:
    """Largest eigenvalue of a small real symmetric PSD matrix."""
    if mat.size == 0:
        return float(floor)
    mat = 0.5 * (mat + mat.T)
    try:
        val = float(np.linalg.eigvalsh(mat)[-1])
    except np.linalg.LinAlgError:
        val = float(np.linalg.norm(mat, ord=2))
    return max(val, float(floor))


def _paper_lipschitz_constants(
    h1: np.ndarray,
    h2: np.ndarray,
    lam: np.ndarray,
    v: np.ndarray,
    mu: np.ndarray,
    num_users: int,
    antennas_per_tx: int,
    rho: float,
    floor: float = 1e-12,
) -> tuple[float, float, float]:
    """Compute Lipschitz constants for the three paper dual-gradient blocks.

    The ICASSP paper states that the three updates use Lipschitz constants
    L_lambda, L_v and L_mu. It does not print closed-form expressions in the
    conference version, so these are the explicit constants implied by the
    regularized-dual gradients and the closed-form primal minimizer.

    Let D = rho I + sum_n mu_n A_n. Since A_n are per-antenna selectors, D is
    diagonal and d_inv is diag(D^{-1}).

    For lambda:
        grad_lambda = H1 x + t 1
    and the Jacobian wrt lambda is
        -0.5 H1 D^{-1} H1^T - (1/(2 rho)) 11^T.
    Therefore a Lipschitz constant is
        0.5 lambda_max(H1 D^{-1} H1^T + (1/rho) 11^T).

    For v:
        grad_v = H2 x,
        L_v = 0.5 lambda_max(H2 D^{-1} H2^T).

    For mu, using the current local Jacobian:
        grad_mu[n] = sum_{i in antenna n group} x_i^2 - P_n.
    With x_i = -0.5 c_i/(rho+mu_n), its local derivative magnitude is
        0.5 sum_i c_i^2/(rho+mu_n)^3.
    """
    denom = _denom_from_mu(mu, num_users, antennas_per_tx, rho)
    d_inv = 1.0 / np.maximum(denom, floor)

    h1_scaled = h1 * d_inv[None, :]
    ones = np.ones((h1.shape[0], h1.shape[0]), dtype=np.float64)
    l_lam_mat = h1_scaled @ h1.T + (1.0 / max(float(rho), floor)) * ones
    l_lam = 0.5 * _max_eigval_symmetric(l_lam_mat, floor)

    if h2.shape[0] > 0:
        h2_scaled = h2 * d_inv[None, :]
        l_v_mat = h2_scaled @ h2.T
        l_v = 0.5 * _max_eigval_symmetric(l_v_mat, floor)
    else:
        l_v = 1.0

    c = h1.T @ lam + h2.T @ v
    ntot = mu.size
    l_mu_vals = np.zeros(ntot, dtype=np.float64)
    for n in range(ntot):
        s = 0.0
        for k in range(num_users):
            off = k * 2 * ntot
            idx_re = off + n
            idx_im = off + ntot + n
            denom_n = max(float(rho + mu[n]), floor)
            s += c[idx_re] ** 2 + c[idx_im] ** 2
        l_mu_vals[n] = 0.5 * s / (denom_n ** 3)
    l_mu = max(float(np.max(l_mu_vals)) if l_mu_vals.size else 0.0, floor)

    return l_lam, l_v, l_mu



def _projected_step_norm(
    lam: np.ndarray,
    v: np.ndarray,
    mu: np.ndarray,
    lam_new: np.ndarray,
    v_new: np.ndarray,
    mu_new: np.ndarray,
) -> float:
    """Norm of the accepted dual update."""
    return float(np.sqrt(
        np.sum((lam_new - lam) ** 2)
        + np.sum((v_new - v) ** 2)
        + np.sum((mu_new - mu) ** 2)
    ))



def _paper_rho_from_beta(
    num_tx_antennas: int,
    num_users: int,
    per_antenna_powers: np.ndarray,
    beta: float,
) -> float:
    """Paper-inspired rho choice from beta/error-bound rule.

    The ICASSP paper defines rho = M alpha / ((N^2 + M N) P) and
    alpha = beta * N * sqrt(P/M) for the equal-P PAPC case.

    For unequal per-antenna limits, use the mean per-antenna power as the
    scalar P in that expression. This keeps the scaling close to the paper
    while allowing unequal TX powers split across antennas.
    """
    n_ant = int(num_tx_antennas)
    m_users = int(num_users)
    p_ref = float(np.mean(per_antenna_powers))
    p_ref = max(p_ref, 1e-12)
    beta = max(float(beta), 1e-12)
    alpha = beta * n_ant * np.sqrt(p_ref / max(m_users, 1))
    return float(m_users * alpha / (((n_ant ** 2) + m_users * n_ant) * p_ref))


def build_low_complexity_papc_zf_precoder(
    g_rows: np.ndarray,
    tx_powers: np.ndarray,
    antennas_per_tx: int,
    *,
    num_iters: int = 1000,
    rho: float = -1.0,
    beta: float = 0.3,
    tol: float = 1e-5,
    lipschitz_floor: float = 1e-12,
    final_safety_backoff: bool = False,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, float]]:
    """Low-complexity regularized-dual ZF solver with per-antenna PAPC.

    This version is aligned with the paper's stated algorithm:

      * per-antenna constraints x^T A_n x <= P_n;
      * Tikhonov-regularized Lagrangian;
      * closed-form minimizer over (t, x);
      * projected dual-gradient updates with separate Lipschitz constants
        L_lambda, L_v and L_mu, matching the update structure in equation (22).

    For D-MIMO TX powers, each TX's total power is split equally among its
    antennas:
        P_n = P_b / antennas_per_tx,  for antennas n belonging to TX b.

    Problem:
        maximize t
        s.t. Re{g_k^H w_k} >= t
             Im{g_k^H w_k} = 0
             g_j^H w_k = 0, j != k
             sum_k |w_k[n]|^2 <= P_n, for every antenna n.
    """
    num_users, ntot = g_rows.shape
    num_tx = tx_powers.size
    per_ant_limits = per_antenna_powers_from_tx_powers(tx_powers, antennas_per_tx)
    if per_ant_limits.size != ntot:
        raise ValueError(
            f"per-antenna limits size {per_ant_limits.size} does not match channel antennas {ntot}."
        )

    rho_eff = float(rho)
    if rho_eff <= 0.0:
        rho_eff = _paper_rho_from_beta(ntot, num_users, per_ant_limits, beta)
    rho_eff = max(rho_eff, 1e-12)
    lipschitz_floor = max(float(lipschitz_floor), 1e-18)

    h1, h2 = _build_real_constraint_matrices(g_rows)

    # Paper-compatible dual variables:
    # lambda >= 0 for desired-gain inequalities,
    # v free for ZF equalities,
    # mu >= 0 for per-antenna power constraints.
    lam = np.ones(num_users, dtype=np.float64) / max(num_users, 1)
    v = np.zeros(h2.shape[0], dtype=np.float64)
    mu = np.zeros(ntot, dtype=np.float64)

    t_val, x = _regularized_dual_primal_minimizer(
        h1, h2, lam, v, mu, num_users, antennas_per_tx, rho_eff
    )
    grad_lam = h1 @ x + t_val
    grad_v = h2 @ x
    antenna_powers = _per_antenna_powers_from_x(x, num_users, ntot)
    grad_mu = antenna_powers - per_ant_limits

    converged = False
    last_step_norm = np.inf
    l_lam = l_v = l_mu = np.nan

    for it in range(int(num_iters)):
        l_lam, l_v, l_mu = _paper_lipschitz_constants(
            h1, h2, lam, v, mu,
            num_users, antennas_per_tx, rho_eff,
            floor=lipschitz_floor,
        )

        lam_new, v_new, mu_new = _dual_projected_lipschitz_step(
            lam, v, mu,
            grad_lam, grad_v, grad_mu,
            l_lam, l_v, l_mu,
            floor=lipschitz_floor,
        )
        last_step_norm = _projected_step_norm(lam, v, mu, lam_new, v_new, mu_new)
        lam, v, mu = lam_new, v_new, mu_new

        t_val, x = _regularized_dual_primal_minimizer(
            h1, h2, lam, v, mu, num_users, antennas_per_tx, rho_eff
        )
        grad_lam = h1 @ x + t_val
        grad_v = h2 @ x
        antenna_powers = _per_antenna_powers_from_x(x, num_users, ntot)
        grad_mu = antenna_powers - per_ant_limits

        eq_residual = float(np.linalg.norm(h2 @ x))
        desired_viol = float(np.max(h1 @ x + t_val))
        power_viol = float(np.max(antenna_powers / np.maximum(per_ant_limits, 1e-12)) - 1.0)

        if (
            last_step_norm <= tol * max(1.0, np.sqrt(lam.size + v.size + mu.size))
            and eq_residual <= 10.0 * tol
            and desired_viol <= 10.0 * tol
            and power_viol <= 10.0 * tol
        ):
            converged = True
            break

    p_raw = _x_to_precoder(x, ntot, num_users)
    raw_ant_powers = per_antenna_powers_from_precoder(p_raw)
    raw_block_powers = per_tx_block_powers_from_precoder(p_raw, num_tx, antennas_per_tx)

    raw_ant_power_ratio = float(np.max(raw_ant_powers / np.maximum(per_ant_limits, 1e-12)))
    if final_safety_backoff and raw_ant_power_ratio > 1.0:
        p_final, alpha, _raw_ant_used_before_scaling = common_per_antenna_backoff(
            p_raw, per_ant_limits
        )
        final_ant_powers = per_antenna_powers_from_precoder(p_final)
    else:
        p_final = p_raw
        alpha = 1.0
        final_ant_powers = raw_ant_powers

    final_block_powers = per_tx_block_powers_from_precoder(p_final, num_tx, antennas_per_tx)

    gp = g_rows @ p_final
    off_diag = gp - np.diag(np.diag(gp))
    desired = np.diag(gp)
    desired_norm = float(np.linalg.norm(desired))
    offdiag_norm = float(np.linalg.norm(off_diag, ord="fro"))
    normalized_leakage = offdiag_norm / max(desired_norm, 1e-12)

    # Regularized dual value for diagnostics only.
    d_val, _, _, _, _, _, _ = _regularized_dual_value_and_gradient(
        h1, h2, lam, v, mu, per_ant_limits,
        num_users, antennas_per_tx, rho_eff,
    )

    info = {
        "t_last": float(t_val),
        "rho_used": float(rho_eff),
        "dual_value": float(d_val),
        "dual_step_norm": float(last_step_norm),
        "converged": float(converged),
        "num_iters_used": float(it + 1 if int(num_iters) > 0 else 0),
        "num_backtracks": 0.0,
        "l_lambda": float(l_lam),
        "l_v": float(l_v),
        "l_mu": float(l_mu),
        "raw_power_violation": raw_ant_power_ratio,
        "final_power_violation": float(np.max(final_ant_powers / np.maximum(per_ant_limits, 1e-12))),
        "raw_tx_power_violation": float(np.max(raw_block_powers / np.maximum(tx_powers, 1e-12))),
        "final_tx_power_violation": float(np.max(final_block_powers / np.maximum(tx_powers, 1e-12))),
        "zf_residual": offdiag_norm,
        "normalized_zf_leakage": normalized_leakage,
        "min_desired_abs": float(np.min(np.abs(desired))) if desired.size else 0.0,
        "sum_lambda": float(np.sum(lam)),
        "max_mu": float(np.max(mu)) if mu.size else 0.0,
    }
    return p_final, alpha, raw_block_powers, info


def compute_sum_rate_lmmse(
    h_agg: np.ndarray,
    precoder: np.ndarray,
    noise_power: float,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Single-stream-per-UE sum-rate with local effective-channel LMMSE receivers."""
    num_users, nrx, _ntot = h_agg.shape
    eye_rx = np.eye(nrx, dtype=np.complex128)
    sinr = np.zeros(num_users, dtype=np.float64)
    for k in range(num_users):
        g_eff = h_agg[k] @ precoder  # [Nrx, K]
        desired = g_eff[:, k]
        cov = noise_power * eye_rx.copy()
        for j in range(num_users):
            if j == k:
                continue
            gj = g_eff[:, j]
            cov += np.outer(gj, gj.conj())
        try:
            solved = np.linalg.solve(cov, desired)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(cov) @ desired
        sinr[k] = max(float(np.real(np.vdot(desired, solved))), 0.0)
    return float(np.sum(np.log2(1.0 + np.maximum(sinr, eps)))), sinr


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1 or window > x.size:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="valid")




def run(cfg: Config, tx_powers: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate multiple independent drops, each with multiple Sionna time steps.

    A drop means a fresh UE placement/topology. Inside each drop, Sionna generates
    cfg.num_slots time samples with the same topology and UE velocities.
    """
    total_samples = int(cfg.num_drops) * int(cfg.num_slots)

    total_tx_power = float(np.sum(tx_powers))
    noise_power = total_tx_power / (10.0 ** (cfg.snr_db / 10.0))

    rate_naive = np.zeros(total_samples, dtype=np.float64)
    rate_papc = np.zeros(total_samples, dtype=np.float64)
    min_rate_naive = np.zeros(total_samples, dtype=np.float64)
    min_rate_papc = np.zeros(total_samples, dtype=np.float64)
    alpha_naive = np.zeros(total_samples, dtype=np.float64)
    alpha_papc = np.zeros(total_samples, dtype=np.float64)
    pblock_naive_raw = np.zeros((total_samples, cfg.num_tx), dtype=np.float64)
    pblock_papc_raw = np.zeros((total_samples, cfg.num_tx), dtype=np.float64)
    singular_values = np.zeros((total_samples, cfg.num_users), dtype=np.float64)
    papc_zf_residual = np.zeros(total_samples, dtype=np.float64)
    papc_normalized_zf_leakage = np.zeros(total_samples, dtype=np.float64)
    papc_min_desired_abs = np.zeros(total_samples, dtype=np.float64)
    papc_raw_power_violation = np.zeros(total_samples, dtype=np.float64)
    papc_final_power_violation = np.zeros(total_samples, dtype=np.float64)
    papc_iters_used = np.zeros(total_samples, dtype=np.float64)
    papc_converged = np.zeros(total_samples, dtype=np.float64)
    papc_l_lambda = np.zeros(total_samples, dtype=np.float64)
    papc_l_v = np.zeros(total_samples, dtype=np.float64)
    papc_l_mu = np.zeros(total_samples, dtype=np.float64)
    drop_indices = np.zeros(total_samples, dtype=np.int64)
    slot_indices = np.zeros(total_samples, dtype=np.int64)

    all_bs_locs = np.zeros((cfg.num_drops, cfg.num_tx, 3), dtype=np.float64)
    all_ue_locs = np.zeros((cfg.num_drops, cfg.num_users, 3), dtype=np.float64)

    sample_idx = 0
    for drop_idx in range(int(cfg.num_drops)):
        drop_rng = np.random.default_rng(int(cfg.seed) + 7919 * drop_idx)
        channels, bs_locs, ue_locs = simulate_sionna_dmimo_channels(cfg, drop_rng, drop_idx=drop_idx)
        all_bs_locs[drop_idx] = bs_locs
        all_ue_locs[drop_idx] = ue_locs

        for slot_idx in range(cfg.num_slots):
            if (slot_idx + 1) % max(1, cfg.num_slots // 20) == 0 or slot_idx == 0:
                print(
                    f"Drop {drop_idx + 1} / {cfg.num_drops}, "
                    f"slot {slot_idx + 1} / {cfg.num_slots}",
                    end="\r",
                )

            h_agg = aggregate_all_users(channels[slot_idx])
            g_rows, _v_right, sigma = dominant_effective_channel_rows(h_agg)
            singular_values[sample_idx] = sigma

            p_naive, a_naive, raw_blocks_naive = build_naive_zf_precoder(
                g_rows, tx_powers, cfg.antennas_per_tx
            )
            p_papc, a_papc, raw_blocks_papc, papc_info = build_low_complexity_papc_zf_precoder(
                g_rows,
                tx_powers,
                cfg.antennas_per_tx,
                num_iters=cfg.papc_num_iters,
                rho=cfg.papc_rho,
                beta=cfg.papc_beta,
                tol=cfg.papc_tol,
                lipschitz_floor=cfg.papc_lipschitz_floor,
                final_safety_backoff=cfg.papc_final_safety_backoff,
            )

            rate_naive[sample_idx], sinr_naive = compute_sum_rate_lmmse(h_agg, p_naive, noise_power)
            rate_papc[sample_idx], sinr_papc = compute_sum_rate_lmmse(h_agg, p_papc, noise_power)

            min_rate_naive[sample_idx] = float(np.min(np.log2(1.0 + np.maximum(sinr_naive, 1e-12))))
            min_rate_papc[sample_idx] = float(np.min(np.log2(1.0 + np.maximum(sinr_papc, 1e-12))))

            alpha_naive[sample_idx] = a_naive
            alpha_papc[sample_idx] = a_papc
            pblock_naive_raw[sample_idx] = raw_blocks_naive
            pblock_papc_raw[sample_idx] = raw_blocks_papc
            papc_zf_residual[sample_idx] = papc_info["zf_residual"]
            papc_normalized_zf_leakage[sample_idx] = papc_info["normalized_zf_leakage"]
            papc_min_desired_abs[sample_idx] = papc_info["min_desired_abs"]
            papc_raw_power_violation[sample_idx] = papc_info["raw_power_violation"]
            papc_final_power_violation[sample_idx] = papc_info["final_power_violation"]
            papc_iters_used[sample_idx] = papc_info["num_iters_used"]
            papc_converged[sample_idx] = papc_info["converged"]
            papc_l_lambda[sample_idx] = papc_info["l_lambda"]
            papc_l_v[sample_idx] = papc_info["l_v"]
            papc_l_mu[sample_idx] = papc_info["l_mu"]
            drop_indices[sample_idx] = drop_idx
            slot_indices[sample_idx] = slot_idx

            sample_idx += 1
        print()

    return {
        "rate_naive": rate_naive,
        "rate_papc": rate_papc,
        "min_rate_naive": min_rate_naive,
        "min_rate_papc": min_rate_papc,
        "alpha_naive": alpha_naive,
        "alpha_papc": alpha_papc,
        "pblock_naive_raw": pblock_naive_raw,
        "pblock_papc_raw": pblock_papc_raw,
        "singular_values": singular_values,
        "papc_zf_residual": papc_zf_residual,
        "papc_normalized_zf_leakage": papc_normalized_zf_leakage,
        "papc_min_desired_abs": papc_min_desired_abs,
        "papc_raw_power_violation": papc_raw_power_violation,
        "papc_final_power_violation": papc_final_power_violation,
        "papc_iters_used": papc_iters_used,
        "papc_converged": papc_converged,
        "papc_l_lambda": papc_l_lambda,
        "papc_l_v": papc_l_v,
        "papc_l_mu": papc_l_mu,
        "drop_indices": drop_indices,
        "slot_indices": slot_indices,
        "bs_locs": all_bs_locs,
        "ue_locs": all_ue_locs,
        "tx_powers": tx_powers,
    }



def save_placement_plot(cfg: Config, results: dict[str, np.ndarray]) -> None:
    """Save a 2D plot of distributed TX and UE locations across drops."""
    bs_locs = np.asarray(results["bs_locs"], dtype=np.float64)
    ue_locs = np.asarray(results["ue_locs"], dtype=np.float64)
    tx_powers = np.asarray(results["tx_powers"], dtype=np.float64)

    # Shapes are [D, B, 3] and [D, K, 3]. Keep backward compatibility with
    # older single-drop files if needed.
    if bs_locs.ndim == 2:
        bs_locs = bs_locs[None, ...]
    if ue_locs.ndim == 2:
        ue_locs = ue_locs[None, ...]

    tx_xy = bs_locs[0, :, :2]
    ue_xy_all = ue_locs[:, :, :2].reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    outer = plt.Circle((0.0, 0.0), float(cfg.cell_radius_m), fill=False, linestyle="--", alpha=0.45)
    inner = plt.Circle((0.0, 0.0), float(cfg.min_ue_distance_m), fill=False, linestyle=":", alpha=0.45)
    ax.add_patch(outer)
    ax.add_patch(inner)

    ax.scatter(tx_xy[:, 0], tx_xy[:, 1], marker="^", s=150, label="Transmitters")
    ax.scatter(ue_xy_all[:, 0], ue_xy_all[:, 1], marker="o", s=35, alpha=0.45, label="UEs across drops")

    for b, (x_b, y_b) in enumerate(tx_xy):
        ax.annotate(
            f"TX {b}\nP={tx_powers[b]:g}\nz={bs_locs[0, b, 2]:.1f} m",
            xy=(x_b, y_b),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
        )

    all_xy = np.vstack([
        tx_xy,
        ue_xy_all,
        [[cfg.cell_radius_m, 0.0], [-cfg.cell_radius_m, 0.0],
         [0.0, cfg.cell_radius_m], [0.0, -cfg.cell_radius_m]],
    ])
    span = max(float(np.ptp(all_xy[:, 0])), float(np.ptp(all_xy[:, 1])), 1.0)
    pad = 0.10 * span
    ax.set_xlim(float(np.min(all_xy[:, 0]) - pad), float(np.max(all_xy[:, 0]) + pad))
    ax.set_ylim(float(np.min(all_xy[:, 1]) - pad), float(np.max(all_xy[:, 1]) + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Sionna {cfg.sionna_scenario.upper()} D-MIMO placements across {bs_locs.shape[0]} drops")
    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(cfg.output_dir / "tx_ue_placements_all_drops.png", dpi=160)
    plt.close(fig)



def empirical_cdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted x values and empirical CDF probabilities."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0]), np.array([0.0])
    xs = np.sort(x)
    ps = np.arange(1, xs.size + 1, dtype=np.float64) / float(xs.size)
    return xs, ps


def plot_two_cdfs(
    values_a: np.ndarray,
    values_b: np.ndarray,
    label_a: str,
    label_b: str,
    xlabel: str,
    title: str,
    output_path: Path,
) -> None:
    xa, pa = empirical_cdf(values_a)
    xb, pb = empirical_cdf(values_b)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(xa, pa, lw=1.7, label=label_a)
    ax.plot(xb, pb, lw=1.7, label=label_b)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_one_cdf(
    values: np.ndarray,
    xlabel: str,
    title: str,
    output_path: Path,
    reference_x: float | None = None,
    reference_label: str | None = None,
) -> None:
    xs, ps = empirical_cdf(values)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(xs, ps, lw=1.7)
    if reference_x is not None:
        ax.axvline(float(reference_x), lw=1.0, linestyle="--", label=reference_label or f"{reference_x:g}")
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_outputs(cfg: Config, results: dict[str, np.ndarray]) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.output_dir / "low_complexity_papc_zf_results.npz", **results)
    save_placement_plot(cfg, results)

    r_naive = np.asarray(results["rate_naive"], dtype=np.float64)
    r_papc = np.asarray(results["rate_papc"], dtype=np.float64)
    min_naive = np.asarray(results["min_rate_naive"], dtype=np.float64)
    min_papc = np.asarray(results["min_rate_papc"], dtype=np.float64)

    plot_two_cdfs(
        r_naive,
        r_papc,
        "Naive ZF + common power backoff",
        "Low-complexity PAPC-ZF",
        "Sum-rate [bits/s/Hz]",
        f"CDF of sum-rate over {cfg.num_drops} drops × {cfg.num_slots} slots/drop",
        cfg.output_dir / "cdf_sum_rate_naive_zf_vs_low_complexity_papc_zf.png",
    )

    plot_two_cdfs(
        min_naive,
        min_papc,
        "Naive ZF + common power backoff",
        "Low-complexity PAPC-ZF",
        "Minimum user rate [bits/s/Hz]",
        f"CDF of minimum user rate over {cfg.num_drops} drops × {cfg.num_slots} slots/drop",
        cfg.output_dir / "cdf_min_rate_naive_zf_vs_low_complexity_papc_zf.png",
    )

    rate_delta = r_papc - r_naive
    plot_one_cdf(
        rate_delta,
        "PAPC-ZF minus naive ZF sum-rate [bits/s/Hz]",
        "CDF of PAPC-ZF sum-rate delta over naive ZF",
        cfg.output_dir / "cdf_sum_rate_delta_low_complexity_papc_zf.png",
        reference_x=0.0,
        reference_label="No gain",
    )

    rate_gain_percent = 100.0 * rate_delta / np.maximum(np.abs(r_naive), 1e-12)
    plot_one_cdf(
        rate_gain_percent,
        "PAPC-ZF sum-rate gain over naive ZF [%]",
        "CDF of PAPC-ZF percentage sum-rate gain",
        cfg.output_dir / "cdf_sum_rate_gain_low_complexity_papc_zf_percent.png",
        reference_x=0.0,
        reference_label="No gain",
    )

    leakage = np.asarray(results["papc_normalized_zf_leakage"], dtype=np.float64)
    plot_one_cdf(
        leakage,
        r"$\|G P - \mathrm{diag}(G P)\|_F / (\|\mathrm{diag}(G P)\|_F + \epsilon)$",
        "CDF of normalized effective-channel leakage",
        cfg.output_dir / "cdf_papc_normalized_zf_leakage.png",
        reference_x=0.1,
        reference_label="0.1 amplitude ratio",
    )

    leakage_db = 20.0 * np.log10(np.maximum(leakage, 1e-12))
    plot_one_cdf(
        leakage_db,
        "Normalized leakage [dB]",
        "CDF of normalized effective-channel leakage",
        cfg.output_dir / "cdf_papc_normalized_zf_leakage_db.png",
        reference_x=-20.0,
        reference_label="-20 dB",
    )

    plot_one_cdf(
        np.asarray(results["papc_final_power_violation"], dtype=np.float64),
        r"$\max_n P_n^{used}/P_n^{max}$",
        "CDF of PAPC-ZF per-antenna power ratio",
        cfg.output_dir / "cdf_papc_power_constraint_ratio.png",
        reference_x=1.0,
        reference_label="Power limit",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-drops", type=int, default=20)
    parser.add_argument("--num-slots", type=int, default=50)
    parser.add_argument("--num-tx", type=int, default=3)
    parser.add_argument("--antennas-per-tx", type=int, default=4)
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--rx-antennas-per-user", type=int, default=2)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--tx-powers", type=str, default="1.0,0.5,0.25")
    parser.add_argument("--sionna-scenario", type=str, default="uma", choices=["umi", "uma", "rma"])
    parser.add_argument("--carrier-frequency-hz", type=float, default=3.5e9)
    parser.add_argument("--subcarrier-spacing-hz", type=float, default=30e3)
    parser.add_argument("--slot-duration-s", type=float, default=1e-3)
    parser.add_argument("--ue-speed-kmh", type=float, default=10.0)
    parser.add_argument("--bs-height-m", type=float, default=25.0)
    parser.add_argument("--ue-height-m", type=float, default=1.5)
    parser.add_argument("--cell-radius-m", type=float, default=100.0)
    parser.add_argument("--min-ue-distance-m", type=float, default=10.0)
    parser.add_argument("--tx-spacing-m", type=float, default=50.0)
    parser.add_argument("--sionna-o2i-model", type=str, default="low")
    parser.add_argument("--disable-pathloss", action="store_true")
    parser.add_argument("--enable-shadow-fading", action="store_true")
    parser.add_argument("--disable-average-channel-normalization", action="store_true")
    parser.add_argument("--moving-avg-window", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--papc-num-iters", type=int, default=50)
    parser.add_argument("--papc-rho", type=float, default=-1.0)
    parser.add_argument("--papc-beta", type=float, default=0.1)
    parser.add_argument("--papc-tol", type=float, default=1e-5)
    parser.add_argument("--papc-lipschitz-floor", type=float, default=1e-12)
    parser.add_argument("--papc-final-safety-backoff", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results_low_complexity_papc_zf"))
    args = parser.parse_args()

    cfg = Config(
        num_drops=args.num_drops,
        num_slots=args.num_slots,
        num_tx=args.num_tx,
        antennas_per_tx=args.antennas_per_tx,
        num_users=args.num_users,
        rx_antennas_per_user=args.rx_antennas_per_user,
        snr_db=args.snr_db,
        seed=args.seed,
        sionna_scenario=args.sionna_scenario,
        carrier_frequency_hz=args.carrier_frequency_hz,
        subcarrier_spacing_hz=args.subcarrier_spacing_hz,
        slot_duration_s=args.slot_duration_s,
        ue_speed_kmh=args.ue_speed_kmh,
        bs_height_m=args.bs_height_m,
        ue_height_m=args.ue_height_m,
        cell_radius_m=args.cell_radius_m,
        min_ue_distance_m=args.min_ue_distance_m,
        tx_spacing_m=args.tx_spacing_m,
        sionna_o2i_model=args.sionna_o2i_model,
        sionna_enable_pathloss=not args.disable_pathloss,
        sionna_enable_shadow_fading=bool(args.enable_shadow_fading),
        normalize_average_channel_power=not args.disable_average_channel_normalization,
        moving_avg_window=args.moving_avg_window,
        papc_num_iters=args.papc_num_iters,
        papc_rho=args.papc_rho,
        papc_beta=args.papc_beta,
        papc_tol=args.papc_tol,
        papc_lipschitz_floor=args.papc_lipschitz_floor,
        papc_final_safety_backoff=bool(args.papc_final_safety_backoff),
        output_dir=args.output_dir,
    )
    tx_powers = parse_tx_powers(args.tx_powers, cfg.num_tx)
    results = run(cfg, tx_powers)
    save_outputs(cfg, results)

    print("Low-complexity PAPC-ZF Sionna D-MIMO demo finished.")
    print(f"TX powers                      : {tx_powers}")
    print(f"Sionna scenario                 : {cfg.sionna_scenario}")
    print(f"UE speed                        : {cfg.ue_speed_kmh:.2f} km/h")
    print(f"Mean naive ZF sum-rate          : {np.mean(results['rate_naive']):.4f} bits/s/Hz")
    print(f"Mean PAPC-ZF sum-rate           : {np.mean(results['rate_papc']):.4f} bits/s/Hz")
    print(f"Mean PAPC-ZF - naive delta      : {np.mean(results['rate_papc'] - results['rate_naive']):.4f} bits/s/Hz")
    print(f"Drops × slots/drop              : {cfg.num_drops} × {cfg.num_slots}")
    print(f"Sum-rate CDF saved to           : {cfg.output_dir / 'cdf_sum_rate_naive_zf_vs_low_complexity_papc_zf.png'}")
    print(f"Placement plot saved to         : {cfg.output_dir / 'tx_ue_placements_all_drops.png'}")
    print(f"Min-rate CDF saved to           : {cfg.output_dir / 'cdf_min_rate_naive_zf_vs_low_complexity_papc_zf.png'}")
    print(f"Leakage CDF saved to            : {cfg.output_dir / 'cdf_papc_normalized_zf_leakage.png'}")
    print(f"Leakage dB CDF saved to         : {cfg.output_dir / 'cdf_papc_normalized_zf_leakage_db.png'}")
    print(f"Power ratio CDF saved to        : {cfg.output_dir / 'cdf_papc_power_constraint_ratio.png'}")
    print(f"Mean PAPC normalized leakage    : {np.mean(results['papc_normalized_zf_leakage']):.4e}")
    print(f"Mean PAPC normalized leakage dB : {20.0 * np.log10(max(np.mean(results['papc_normalized_zf_leakage']), 1e-12)):.2f} dB")
    print(f"Mean PAPC raw ant-power ratio   : {np.mean(results['papc_raw_power_violation']):.4f}")
    print(f"Mean PAPC final ant-power ratio : {np.mean(results['papc_final_power_violation']):.4f}")
    print(f"PAPC convergence fraction       : {np.mean(results['papc_converged']):.4f}")
    print(f"Results saved to                : {cfg.output_dir / 'low_complexity_papc_zf_results.npz'}")


if __name__ == "__main__":
    main()