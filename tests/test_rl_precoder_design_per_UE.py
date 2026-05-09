from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class SimConfig:
    num_tx_antennas: int = 4
    num_users: int = 2
    num_rx_antennas_per_user: int = 2
    streams_per_user: int = 1
    num_slots: int = 2000
    snr_db: float = 10.0
    temporal_correlation: float = 0.95
    seed: int = 7
    total_tx_power: float = 1.0


@dataclass
class RLConfig:
    lr_out: float = 3e-2
    fixed_kappa: float = 10.0
    reward_baseline_beta: float = 0.99
    advantage_clip: float = 1.0
    grad_clip_norm: float = 1.0
    init_scale_out: float = 1e-2
    batch_size: int = 512
    reservoir_size: int = 128
    spectral_radius: float = 0.8
    input_scale: float = 0.15
    # Deprecated/unused for SLNR-ratio shaping; denominator now uses noise power.
    leakage_lambda: float = 1.0
    signal_gamma: float = 0.5
    max_fb_resamples: int = 16
    leakage_norm_eps: float = 1e-12
    signal_norm_eps: float = 1e-12
    reward_mode: str = "actual_sinr_log_ratio"
    reward_sinr_eps: float = 1e-12
    best_of_n: int = 8


def complex_gaussian(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def normalize_columns_equal_power(precoder: np.ndarray, total_tx_power: float) -> np.ndarray:
    k = precoder.shape[1]
    p_col = total_tx_power / k
    out = np.zeros_like(precoder)
    for i in range(k):
        col = precoder[:, i]
        nrm = np.linalg.norm(col)
        if nrm < 1e-12:
            col = np.ones_like(col) / np.sqrt(col.size)
            nrm = np.linalg.norm(col)
        out[:, i] = np.sqrt(p_col) * (col / nrm)
    return out


def build_zf_precoder_from_pmi(
    full_vk_list: list[np.ndarray], streams_per_user: int, total_tx_power: float
) -> np.ndarray:
    selected_rows = []
    for vk in full_vk_list:
        selected_rows.append(vk[:, :streams_per_user].conj().T)
    z_matrix = np.vstack(selected_rows)
    return normalize_columns_equal_power(np.linalg.pinv(z_matrix), total_tx_power)


def compute_slot_sum_rate(
    user_channels: np.ndarray, precoder: np.ndarray, noise_power: float
) -> tuple[float, np.ndarray]:
    num_users = user_channels.shape[0]
    sinr = np.zeros(num_users, dtype=np.float64)

    for k in range(num_users):
        hk = user_channels[k]
        signal_vec = hk @ precoder[:, k]
        uk = signal_vec / max(np.linalg.norm(signal_vec), 1e-12)

        desired = np.abs(np.vdot(uk, hk @ precoder[:, k])) ** 2
        interference = 0.0
        for j in range(num_users):
            if j != k:
                interference += np.abs(np.vdot(uk, hk @ precoder[:, j])) ** 2
        sinr[k] = desired / (interference + noise_power)

    return float(np.sum(np.log2(1.0 + sinr))), sinr



def compute_slot_sinr_proxy(
    user_channels: np.ndarray, precoder: np.ndarray, noise_power: float
) -> np.ndarray:
    """Compute a simple multi-user SINR proxy for reward shaping.

    This proxy uses the total received energy from each beam at each UE:

        desired_k = ||H_k p_k||_2^2
        interference_k = sum_{j != k} ||H_k p_j||_2^2
        proxy_sinr_k = desired_k / (interference_k + noise_power)

    It is different from compute_slot_sum_rate(), which first forms a receive
    combiner along the desired received vector and then projects interference
    through that combiner.  The proxy is intentionally simpler and gives the RL
    update a user-wise, SINR-like signal rather than only a scalar sum-rate
    difference.
    """
    num_users = user_channels.shape[0]
    proxy_sinr = np.zeros(num_users, dtype=np.float64)

    for k in range(num_users):
        hk = user_channels[k]
        desired = np.linalg.norm(hk @ precoder[:, k]) ** 2
        interference = 0.0
        for j in range(num_users):
            if j != k:
                interference += np.linalg.norm(hk @ precoder[:, j]) ** 2
        proxy_sinr[k] = desired / (interference + noise_power)

    return proxy_sinr


def compute_rl_reward(
    reward_mode: str,
    rate: float,
    zf_rate: float,
    actual_sinr: np.ndarray,
    zf_actual_sinr: np.ndarray,
    proxy_sinr: np.ndarray,
    zf_proxy_sinr: np.ndarray,
    eps: float,
) -> float:
    """Compute the scalar RL reward used in REINFORCE.

    Supported modes:
      - throughput_delta: old reward, R_RL - R_ZF.
      - normalized_throughput_delta: (R_RL - R_ZF) / (|R_ZF| + eps).
      - actual_sinr_log_ratio: mean_k log((SINR_RL,k + eps)/(SINR_ZF,k + eps)).
      - sinr_proxy_log_ratio: same log-ratio but using the simpler SINR proxy.

    The default is actual_sinr_log_ratio so the policy is trained using the
    true post-detection SINR returned by compute_slot_sum_rate().
    """
    if reward_mode == "throughput_delta":
        return float(rate - zf_rate)
    if reward_mode == "normalized_throughput_delta":
        return float((rate - zf_rate) / (abs(zf_rate) + eps))
    if reward_mode == "actual_sinr_log_ratio":
        return float(np.mean(np.log((actual_sinr + eps) / (zf_actual_sinr + eps))))
    if reward_mode == "sinr_proxy_log_ratio":
        return float(np.mean(np.log((proxy_sinr + eps) / (zf_proxy_sinr + eps))))
    raise ValueError(
        f"Unknown reward_mode={reward_mode!r}. Expected one of: "
        "throughput_delta, normalized_throughput_delta, "
        "actual_sinr_log_ratio, sinr_proxy_log_ratio."
    )

def pmi_features_from_channels(user_channels: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    vk_list: list[np.ndarray] = []
    features = []
    for k in range(user_channels.shape[0]):
        _, _, vh = np.linalg.svd(user_channels[k], full_matrices=True)
        vk = vh.conj().T
        vk_list.append(vk)
        features += [np.real(vk).reshape(-1), np.imag(vk).reshape(-1)]
    return np.concatenate(features, axis=0).astype(np.float64), vk_list


def simulate_channels(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    channels = np.zeros(
        (cfg.num_slots, cfg.num_users, cfg.num_rx_antennas_per_user, cfg.num_tx_antennas),
        dtype=np.complex128,
    )
    h_prev = complex_gaussian(
        (cfg.num_users, cfg.num_rx_antennas_per_user, cfg.num_tx_antennas), rng
    )
    for t in range(cfg.num_slots):
        innovation = complex_gaussian(h_prev.shape, rng)
        h_prev = cfg.temporal_correlation * h_prev + np.sqrt(
            1.0 - cfg.temporal_correlation**2
        ) * innovation
        channels[t] = h_prev
    return channels


def run_zf_baseline(cfg: SimConfig, channels: np.ndarray) -> dict[str, np.ndarray]:
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    sinr_trace = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.float64)
    precoders = np.zeros(
        (cfg.num_slots, cfg.num_tx_antennas, cfg.num_users), dtype=np.complex128
    )
    pmi_features = []

    for t in range(cfg.num_slots):
        print(f"ZF Slot {t + 1} / {cfg.num_slots}", end="\r")
        feat, vk_list = pmi_features_from_channels(channels[t])
        pmi_features.append(feat)
        p_zf = build_zf_precoder_from_pmi(
            vk_list, cfg.streams_per_user, cfg.total_tx_power
        )
        throughput[t], sinr_trace[t] = compute_slot_sum_rate(channels[t], p_zf, noise_power)
        precoders[t] = p_zf

    print()
    return {
        "throughput": throughput,
        "sinr": sinr_trace,
        "precoders": precoders,
        "pmi_features": np.stack(pmi_features, axis=0),
    }


def unit_norm(x: np.ndarray) -> np.ndarray:
    return x / max(np.linalg.norm(x), 1e-12)


def real_to_complex_beam(x: np.ndarray, total_tx_power: float, num_users: int) -> np.ndarray:
    nt = x.size // 2
    return np.sqrt(total_tx_power / num_users) * (x[:nt] + 1j * x[nt:])


def beam_similarity(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    sims = []
    for k in range(p_a.shape[1]):
        num = np.abs(np.vdot(p_a[:, k], p_b[:, k]))
        den = np.linalg.norm(p_a[:, k]) * np.linalg.norm(p_b[:, k])
        sims.append(num / max(den, 1e-12))
    return np.array(sims, dtype=np.float64)




def complex_hermitian_to_real_quadratic(a_complex: np.ndarray) -> np.ndarray:
    """Return real matrix A_r such that v^H A v = x^T A_r x.

    Here v = a + j b and x = [a^T, b^T]^T.  For Hermitian
    A = H^H H, the resulting real matrix is symmetric PSD.
    """
    a_re = np.real(a_complex)
    a_im = np.imag(a_complex)
    return np.block([[a_re, -a_im], [a_im, a_re]]).astype(np.float64)


def build_signal_and_leakage_matrices_real(user_channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build per-user real desired-signal and leakage matrices.

    G_k = real_rep(H_k^H H_k), so x_k^T G_k x_k is proportional
    to the useful channel gain of beam k for UE k.

    L_k = sum_{j != k} real_rep(H_j^H H_j), so x_k^T L_k x_k is
    proportional to the leakage caused by beam k into all other UEs.
    """
    num_users, _, num_tx_antennas = user_channels.shape
    d = 2 * num_tx_antennas
    signal_mats = np.zeros((num_users, d, d), dtype=np.float64)
    leakage_mats = np.zeros((num_users, d, d), dtype=np.float64)

    gram_real = []
    for j in range(num_users):
        h_j = user_channels[j]
        gram_j = h_j.conj().T @ h_j
        gram_j_real = complex_hermitian_to_real_quadratic(gram_j)
        gram_j_real = 0.5 * (gram_j_real + gram_j_real.T)
        gram_real.append(gram_j_real)

    for k in range(num_users):
        signal_mats[k] = gram_real[k]
        for j in range(num_users):
            if j != k:
                leakage_mats[k] += gram_real[j]
        leakage_mats[k] = 0.5 * (leakage_mats[k] + leakage_mats[k].T)

    return signal_mats, leakage_mats


def normalized_leakage_score(x: np.ndarray, leakage_mat: np.ndarray, eps: float = 1e-12) -> float:
    """Dimensionless leakage score used in exp(-lambda * leakage).

    The normalization by trace(L)/d makes lambda easier to tune across
    slots and channel realizations.  A score around 1 means roughly
    average leakage over random unit directions.
    """
    d = x.size
    raw = float(x @ leakage_mat @ x)
    scale = float(np.trace(leakage_mat) / max(d, 1))
    return raw / max(scale, eps)




def normalized_signal_score(x: np.ndarray, signal_mat: np.ndarray, eps: float = 1e-12) -> float:
    """Dimensionless desired-signal gain score.

    The normalization by trace(G)/d makes the score roughly equal to 1
    for an average random unit direction. Larger values mean stronger
    useful gain for the intended UE.
    """
    d = x.size
    raw = float(x @ signal_mat @ x)
    scale = float(np.trace(signal_mat) / max(d, 1))
    return raw / max(scale, eps)


def normalized_signal_upper_bound(signal_mat: np.ndarray, eps: float = 1e-12) -> float:
    """Upper bound for normalized x^T G x over unit-norm x.

    max_{||x||=1} x^T G x = lambda_max(G). After trace normalization,
    the bound is lambda_max(G) / (trace(G)/d). This is used to make
    the Fisher-Bingham rejection probability <= 1 when a positive
    desired-signal term is included.
    """
    d = signal_mat.shape[0]
    scale = float(np.trace(signal_mat) / max(d, 1))
    if scale <= eps:
        return 0.0
    # eigvalsh is appropriate because signal_mat is symmetric PSD.
    lam_max = float(np.linalg.eigvalsh(signal_mat).max())
    return lam_max / max(scale, eps)


def raw_quadratic_score(x: np.ndarray, mat: np.ndarray) -> float:
    """Raw quadratic score x^T A x without trace normalization."""
    return float(x @ mat @ x)


def raw_signal_upper_bound(signal_mat: np.ndarray) -> float:
    """Upper bound for raw x^T G x over unit-norm x.

    max_{||x||=1} x^T G x = lambda_max(G).  For the PMI-only
    rank-one projector case, this is approximately 1.
    """
    return float(np.linalg.eigvalsh(signal_mat).max())


def pmi_dominant_vectors_from_channels(user_channels: np.ndarray) -> np.ndarray:
    """Return one dominant right singular vector q_k per UE.

    This function emulates PMI/right-singular-vector feedback in this toy
    testbench.  The policy sampler below uses only these q_k vectors for
    desired-signal and leakage shaping, not full channel Gram matrices H_k^H H_k.
    """
    q_list: list[np.ndarray] = []
    for k in range(user_channels.shape[0]):
        _, _, vh = np.linalg.svd(user_channels[k], full_matrices=True)
        q = vh.conj().T[:, 0]
        q_list.append(q / max(np.linalg.norm(q), 1e-12))
    return np.stack(q_list, axis=0)


def build_pmi_signal_and_leakage_matrices_real(q_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build per-UE PMI-only desired and leakage matrices.

    q_vectors[k] is the dominant right singular vector / PMI direction for UE k.

    G_k^PMI = real_rep(q_k q_k^H), so x_k^T G_k^PMI x_k = |q_k^H v_k|^2.

    L_k^PMI = sum_{j != k} real_rep(q_j q_j^H), so x_k^T L_k^PMI x_k
    approximates how much beam k overlaps with other UEs' PMI directions.

    This replaces the earlier full-channel matrices based on H_k^H H_k.
    """
    num_users, num_tx_antennas = q_vectors.shape
    d = 2 * num_tx_antennas
    signal_mats = np.zeros((num_users, d, d), dtype=np.float64)
    leakage_mats = np.zeros((num_users, d, d), dtype=np.float64)

    pmi_gram_real: list[np.ndarray] = []
    for k in range(num_users):
        qk = q_vectors[k]
        gram_k = np.outer(qk, qk.conj())
        gram_k_real = complex_hermitian_to_real_quadratic(gram_k)
        gram_k_real = 0.5 * (gram_k_real + gram_k_real.T)
        pmi_gram_real.append(gram_k_real)

    for k in range(num_users):
        signal_mats[k] = pmi_gram_real[k]
        for j in range(num_users):
            if j != k:
                leakage_mats[k] += pmi_gram_real[j]
        leakage_mats[k] = 0.5 * (leakage_mats[k] + leakage_mats[k].T)

    return signal_mats, leakage_mats


def compute_pmi_sinr_proxy_from_precoder(
    q_vectors: np.ndarray,
    precoder: np.ndarray,
    noise_power: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute a PMI-only SINR proxy from right singular vectors and a precoder.

    q_vectors[k] is the PMI/right singular vector for UE k.  This uses

        desired_k = |q_k^H p_k|^2
        interference_k = sum_{j != k} |q_k^H p_j|^2
        SINR_hat_k = desired_k / (interference_k + noise_power)

    and does not use the full channel matrix H_k.
    """
    num_users = q_vectors.shape[0]
    out = np.zeros(num_users, dtype=np.float64)
    for k in range(num_users):
        qk = q_vectors[k]
        desired = float(np.abs(np.vdot(qk, precoder[:, k])) ** 2)
        interference = 0.0
        for j in range(num_users):
            if j != k:
                interference += float(np.abs(np.vdot(qk, precoder[:, j])) ** 2)
        out[k] = desired / (interference + noise_power + eps)
    return out


def sample_fisher_bingham_signal_leakage(
    mu: np.ndarray,
    kappa: float,
    signal_mat: np.ndarray,
    leakage_mat: np.ndarray,
    signal_gamma: float,
    slnr_noise_power: float,
    rng: np.random.Generator,
    max_resamples: int,
    signal_norm_eps: float,
    leakage_norm_eps: float,
) -> tuple[np.ndarray, float, float, int, bool, np.ndarray]:
    """Sample from a PMI-only actual-SLNR-shaped policy.

    Proposal:
        q(x) = vMF(mu, kappa)

    Target up to proportionality:
        p(x) ∝ exp(kappa mu^T x
                   + signal_gamma * g(x)/(ell(x) + sigma2_eff)).

    This uses the PMI-only version of the actual SLNR form without trace
    normalization:

        g(x)   = x^T G_k^PMI x   = |q_k^H v_k|^2
        ell(x) = x^T L_k^PMI x   = sum_{j != k} |q_j^H v_k|^2

    where x is the real representation of the unit-norm complex beam v_k.
    Since the actual transmitted beam is sqrt(P_user) * v_k, the equivalent
    noise term in this unit-beam SLNR is

        sigma2_eff = noise_power / P_user.

    The acceptance probability uses the bound g(x) <= lambda_max(G_k^PMI)
    and ell(x) >= 0, so the ratio is upper-bounded by
    lambda_max(G_k^PMI) / sigma2_eff.

    If all attempts are rejected, return the candidate with the largest raw
    PMI-only SLNR score.  In addition, return an empirical shaped-distribution
    mean estimate from the same proposal pool.  This mean is used to center the
    REINFORCE score function during training.
    """
    best_x = None
    best_signal = -np.inf
    best_leakage = np.inf
    best_score = -np.inf
    best_attempt = max_resamples

    # Raw, unnormalized upper bound: max_{||x||=1} x^T G x.
    g_upper = raw_signal_upper_bound(signal_mat)
    noise_floor = max(float(slnr_noise_power), leakage_norm_eps)
    shaped_upper = signal_gamma * g_upper / noise_floor

    # Keep the full resampling pool so we can estimate the mean of the
    # SLNR-shaped distribution induced by vMF proposals and the shaping score.
    pool_x: list[np.ndarray] = []
    pool_scores: list[float] = []
    first_accepted: tuple[np.ndarray, float, float, int, bool] | None = None

    for attempt in range(1, max_resamples + 1):
        x = sample_vmf(mu, kappa, rng)
        g = raw_quadratic_score(x, signal_mat)
        ell = raw_quadratic_score(x, leakage_mat)
        shaped_score = signal_gamma * g / max(ell + noise_floor, leakage_norm_eps)

        pool_x.append(x)
        pool_scores.append(shaped_score)

        if shaped_score > best_score:
            best_x = x
            best_signal = g
            best_leakage = ell
            best_score = shaped_score
            best_attempt = attempt

        log_accept_prob = shaped_score - shaped_upper
        # Numerical safety: because of floating-point roundoff, clip to <= 0.
        log_accept_prob = min(0.0, float(log_accept_prob))
        if first_accepted is None and rng.uniform(0.0, 1.0) < np.exp(log_accept_prob):
            # Preserve the original execution rule: use the first accepted
            # sample if one appears.  We still continue drawing the pool to
            # estimate the empirical shaped mean used for the update.
            first_accepted = (x, g, ell, attempt, True)

    assert best_x is not None

    x_pool = np.stack(pool_x, axis=0)
    score_arr = np.array(pool_scores, dtype=np.float64)
    # Stable importance weights proportional to exp(shaped_score).  Since the
    # proposals are from q(x)=vMF(mu,kappa), this approximates the mean under
    # q(x) exp(raw-SLNR shaping).
    score_arr = score_arr - np.max(score_arr)
    weights = np.exp(score_arr)
    weights = weights / max(float(np.sum(weights)), leakage_norm_eps)
    empirical_mean = np.sum(weights[:, None] * x_pool, axis=0)

    if first_accepted is not None:
        x, g, ell, attempt, accepted = first_accepted
        return x, g, ell, attempt, accepted, empirical_mean

    return best_x, best_signal, best_leakage, best_attempt, False, empirical_mean

def _sample_weight_vmf(dim: int, kappa: float, rng: np.random.Generator) -> float:
    b = (-2.0 * kappa + np.sqrt(4.0 * kappa**2 + (dim - 1) ** 2)) / (dim - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (dim - 1) * np.log(1.0 - x0**2)

    while True:
        z = rng.beta((dim - 1) / 2.0, (dim - 1) / 2.0)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = rng.uniform(0.0, 1.0)
        if kappa * w + (dim - 1) * np.log(1.0 - x0 * w) - c >= np.log(u):
            return float(w)


def sample_vmf(mu: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    dim = mu.size
    mu = unit_norm(mu)
    if kappa < 1e-8:
        return unit_norm(rng.standard_normal(dim))

    w = _sample_weight_vmf(dim, kappa, rng)
    v = unit_norm(rng.standard_normal(dim - 1))
    orth = np.concatenate(([w], np.sqrt(max(1.0 - w * w, 0.0)) * v))

    e1 = np.zeros(dim, dtype=np.float64)
    e1[0] = 1.0
    if np.allclose(mu, e1):
        return orth
    if np.allclose(mu, -e1):
        orth[0] *= -1.0
        return orth

    u = unit_norm(e1 - mu)
    return orth - 2.0 * np.dot(u, orth) * u


def run_random_vmf_baseline(
    cfg: SimConfig, channels: np.ndarray, rng: np.random.Generator, fixed_kappa: float
) -> dict[str, np.ndarray]:
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    k = cfg.num_users
    d = 2 * cfg.num_tx_antennas

    for t in range(cfg.num_slots):
        print(f"Random vMF Slot {t + 1} / {cfg.num_slots}", end="\r")
        beams = np.zeros((cfg.num_tx_antennas, k), dtype=np.complex128)
        for ku in range(k):
            mu = unit_norm(rng.standard_normal(d))
            x_sample = sample_vmf(mu, fixed_kappa, rng)
            beams[:, ku] = real_to_complex_beam(x_sample, cfg.total_tx_power, k)
        throughput[t], _ = compute_slot_sum_rate(channels[t], beams, noise_power)

    print()
    return {"throughput": throughput}


def split_tanh_np(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.real(x)) + 1j * np.tanh(np.imag(x))


def make_fixed_esn(
    feat_dim: int, reservoir_size: int, spectral_radius: float, input_scale: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    w_in = input_scale * complex_gaussian((reservoir_size, feat_dim), rng)
    w_res = complex_gaussian((reservoir_size, reservoir_size), rng)
    eigvals = np.linalg.eigvals(w_res)
    w_res = (spectral_radius / max(np.max(np.abs(eigvals)), 1e-12)) * w_res
    return w_in, w_res


def compute_wesn_states(pmi_features: np.ndarray, rl_cfg: RLConfig, rng: np.random.Generator) -> np.ndarray:
    """Compute fixed WESN readout features from all-UE PMI features.

    This uses the same reservoir state update as the vanilla ESN,

        z_t = tanh(W_in y_t + W_res z_{t-1}),

    but uses the WESN/skip-readout feature

        s_t = [Re{z_t}, Im{z_t}, y_t].

    In this script, y_t is already real-valued because pmi_features_from_channels()
    concatenates Re{V_k} and Im{V_k}. Therefore, we append y_t directly rather
    than splitting it again. Only W_out is trained by the RL algorithm.
    """
    num_slots, feat_dim = pmi_features.shape
    nz = rl_cfg.reservoir_size
    w_in, w_res = make_fixed_esn(
        feat_dim=feat_dim,
        reservoir_size=nz,
        spectral_radius=rl_cfg.spectral_radius,
        input_scale=rl_cfg.input_scale,
        rng=rng,
    )

    z = np.zeros(nz, dtype=np.complex128)
    states = np.zeros((num_slots, 2 * nz + feat_dim), dtype=np.float64)
    for t in range(num_slots):
        y = pmi_features[t]
        z = split_tanh_np(w_in @ y + w_res @ z)
        states[t, :] = np.concatenate([np.real(z), np.imag(z), y])
    return states


def vmf_log_prob_fixed_kappa_torch(
    x: torch.Tensor, mu: torch.Tensor, fixed_kappa: float
) -> torch.Tensor:
    """vMF log probability up to a constant when kappa is fixed."""
    return fixed_kappa * torch.sum(mu * x, dim=-1)


def run_esn_policy_rl(
    cfg: SimConfig,
    rl_cfg: RLConfig,
    channels: np.ndarray,
    zf_baseline: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    k = cfg.num_users
    d = 2 * cfg.num_tx_antennas
    # The SLNR sampler uses raw PMI-only quadratic scores for a unit-norm
    # complex beam v_k: |q_k^H v_k|^2 / (sum_j |q_j^H v_k|^2 + sigma2_eff).
    # Since the transmitted beam is sqrt(P_user) * v_k, the equivalent
    # unit-beam noise term is sigma2_eff = noise_power / P_user.
    per_user_tx_power = cfg.total_tx_power / max(k, 1)
    slnr_noise_power = noise_power / max(per_user_tx_power, rl_cfg.reward_sinr_eps)

    # Shared WESN readout feature generated from the PMI feedback of all UEs.
    # The reservoir update is unchanged from the vanilla ESN. The readout feature
    # is augmented with a skip connection from the current PMI input.
    wesn_states = compute_wesn_states(zf_baseline["pmi_features"], rl_cfg, rng)
    state_dim = wesn_states.shape[1]

    # W_out maps shared WESN readout features to all UE vMF mean logits,
    # shape [K, D, 2*N_z + input_dim]. This is the only trainable policy
    # parameter in this fixed-reservoir WESN setup.
    w_out_init = rl_cfg.init_scale_out * rng.standard_normal((k, d, state_dim))
    w_out = torch.nn.Parameter(torch.tensor(w_out_init, dtype=torch.float64))
    optimizer = torch.optim.Adam([w_out], lr=rl_cfg.lr_out)

    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    reward_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    advantage_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    baseline_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    kappa_trace = np.full((cfg.num_slots, k), rl_cfg.fixed_kappa, dtype=np.float64)
    beat_zf_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    beam_similarity_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    grad_norm_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    loss_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    rate_delta_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    proxy_sinr_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    zf_proxy_sinr_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    proxy_sinr_ratio_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    actual_sinr_ratio_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    signal_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    leakage_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    fb_attempts_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    fb_accept_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    best_of_n_score_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    best_of_n_selected_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    empirical_mean_norm_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    centered_update_norm_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)

    reward_baseline: float | None = None
    batch_s: list[np.ndarray] = []
    batch_x: list[np.ndarray] = []
    batch_emp_mean: list[np.ndarray] = []
    batch_adv: list[float] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        nonlocal batch_s, batch_x, batch_emp_mean, batch_adv, batch_indices
        if not batch_s:
            return

        s_batch = torch.tensor(np.stack(batch_s, axis=0), dtype=torch.float64)
        x_batch = torch.tensor(np.stack(batch_x, axis=0), dtype=torch.float64)
        emp_mean_batch = torch.tensor(np.stack(batch_emp_mean, axis=0), dtype=torch.float64)
        adv_batch = torch.tensor(np.array(batch_adv), dtype=torch.float64)

        # Recompute log probabilities under the current W_out for REINFORCE.
        # logits/mu shape: [B, K, D]
        logits = torch.einsum("kdn,bn->bkd", w_out, s_batch)
        mu = logits / torch.clamp(torch.linalg.norm(logits, dim=-1, keepdim=True), min=1e-12)

        # Empirical-mean-corrected score-function surrogate.
        # The executed sample comes from vMF proposals filtered by the raw-SLNR
        # accept/reject mechanism, so the correct shaped-policy score contains
        # a centering term x - E_shaped[x].  emp_mean_batch is a Monte-Carlo
        # estimate of E_shaped[x] from the same resampling pool used by the
        # sampler.  This reduces the mismatch between the actual sampler and the
        # plain-vMF REINFORCE update.
        centered_x = x_batch - emp_mean_batch
        per_user_log_prob = vmf_log_prob_fixed_kappa_torch(centered_x, mu, rl_cfg.fixed_kappa)
        joint_log_prob = torch.sum(per_user_log_prob, dim=1)
        loss = -torch.mean(adv_batch * joint_log_prob)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([w_out], rl_cfg.grad_clip_norm)
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        grad_norm_value = float(grad_norm.detach().cpu().item())
        for idx in batch_indices:
            loss_trace[idx] = loss_value
            grad_norm_trace[idx] = grad_norm_value

        batch_s = []
        batch_x = []
        batch_emp_mean = []
        batch_adv = []
        batch_indices = []

    for t in range(cfg.num_slots):
        print(f"ESN-RL Slot {t + 1} / {cfg.num_slots}", end="\r")
        s = wesn_states[t]

        # All users' vMF means are produced from the same shared WESN readout feature.
        s_t = torch.tensor(s, dtype=torch.float64)
        with torch.no_grad():
            logits = torch.einsum("kdn,n->kd", w_out, s_t)
            mu_t = logits / torch.clamp(torch.linalg.norm(logits, dim=-1, keepdim=True), min=1e-12)
            mu_np = mu_t.detach().cpu().numpy()

        # Emulate the limited-feedback information available to the policy.
        # The sampler uses only dominant right singular vectors / PMI directions,
        # not full channel Gram matrices H_k^H H_k.
        q_vectors = pmi_dominant_vectors_from_channels(channels[t])
        signal_mats, leakage_mats = build_pmi_signal_and_leakage_matrices_real(q_vectors)

        zf_rate = float(zf_baseline["throughput"][t])
        zf_actual_sinr = zf_baseline["sinr"][t]
        zf_proxy_sinr = compute_pmi_sinr_proxy_from_precoder(
            q_vectors, zf_baseline["precoders"][t], noise_power, eps=rl_cfg.reward_sinr_eps
        )

        # Best-of-N full-precoder selection. Each candidate is a complete
        # K-column precoder generated by the current per-UE PMI-only
        # signal/leakage sampler. We score each full candidate using the
        # same scalar reward used for the policy-gradient update and keep
        # the best candidate for execution/training.
        best_score = -np.inf
        best_candidate_index = 0
        best_beams = None
        best_x_sample = None
        best_signal = None
        best_leakage = None
        best_attempts = None
        best_accept = None
        best_emp_mean = None
        best_rate = None
        best_actual_sinr = None
        best_proxy_sinr = None
        best_reward = None

        for cand_idx in range(max(1, rl_cfg.best_of_n)):
            cand_beams = np.zeros((cfg.num_tx_antennas, k), dtype=np.complex128)
            cand_x_sample = np.zeros((k, d), dtype=np.float64)
            cand_signal = np.zeros(k, dtype=np.float64)
            cand_leakage = np.zeros(k, dtype=np.float64)
            cand_attempts = np.zeros(k, dtype=np.float64)
            cand_accept = np.zeros(k, dtype=np.float64)
            cand_emp_mean = np.zeros((k, d), dtype=np.float64)

            for ku in range(k):
                xk, sig, ell, attempts, accepted, emp_mean = sample_fisher_bingham_signal_leakage(
                    mu=mu_np[ku],
                    kappa=rl_cfg.fixed_kappa,
                    signal_mat=signal_mats[ku],
                    leakage_mat=leakage_mats[ku],
                    signal_gamma=rl_cfg.signal_gamma,
                    slnr_noise_power=slnr_noise_power,
                    rng=rng,
                    max_resamples=rl_cfg.max_fb_resamples,
                    signal_norm_eps=rl_cfg.signal_norm_eps,
                    leakage_norm_eps=rl_cfg.leakage_norm_eps,
                )
                cand_x_sample[ku] = xk
                cand_signal[ku] = sig
                cand_leakage[ku] = ell
                cand_attempts[ku] = attempts
                cand_accept[ku] = 1.0 if accepted else 0.0
                cand_emp_mean[ku] = emp_mean
                cand_beams[:, ku] = real_to_complex_beam(cand_x_sample[ku], cfg.total_tx_power, k)

            cand_rate, cand_actual_sinr = compute_slot_sum_rate(
                channels[t], cand_beams, noise_power
            )
            # The RL reward uses the true post-detection SINR when
            # reward_mode="actual_sinr_log_ratio".  The PMI-only proxy is
            # still computed and logged for diagnostics.
            cand_proxy_sinr = compute_pmi_sinr_proxy_from_precoder(
                q_vectors, cand_beams, noise_power, eps=rl_cfg.reward_sinr_eps
            )
            cand_reward = compute_rl_reward(
                reward_mode=rl_cfg.reward_mode,
                rate=cand_rate,
                zf_rate=zf_rate,
                actual_sinr=cand_actual_sinr,
                zf_actual_sinr=zf_actual_sinr,
                proxy_sinr=cand_proxy_sinr,
                zf_proxy_sinr=zf_proxy_sinr,
                eps=rl_cfg.reward_sinr_eps,
            )

            if cand_reward > best_score:
                best_score = cand_reward
                best_candidate_index = cand_idx
                best_beams = cand_beams
                best_x_sample = cand_x_sample
                best_signal = cand_signal
                best_leakage = cand_leakage
                best_attempts = cand_attempts
                best_accept = cand_accept
                best_emp_mean = cand_emp_mean
                best_rate = cand_rate
                best_actual_sinr = cand_actual_sinr
                best_proxy_sinr = cand_proxy_sinr
                best_reward = cand_reward

        assert best_beams is not None
        beams = best_beams
        x_sample = best_x_sample
        signal_trace[t] = best_signal
        leakage_trace[t] = best_leakage
        fb_attempts_trace[t] = best_attempts
        fb_accept_trace[t] = best_accept
        emp_mean_sample = best_emp_mean
        empirical_mean_norm_trace[t] = np.linalg.norm(emp_mean_sample, axis=1)
        centered_update_norm_trace[t] = np.linalg.norm(x_sample - emp_mean_sample, axis=1)
        rate = float(best_rate)
        actual_sinr = best_actual_sinr
        proxy_sinr = best_proxy_sinr
        reward = float(best_reward)
        best_of_n_score_trace[t] = best_score
        best_of_n_selected_trace[t] = best_candidate_index

        if reward_baseline is None:
            reward_baseline = reward
        advantage = reward - reward_baseline
        if rl_cfg.advantage_clip > 0:
            advantage = float(np.clip(advantage, -rl_cfg.advantage_clip, rl_cfg.advantage_clip))

        reward_baseline = (
            rl_cfg.reward_baseline_beta * reward_baseline
            + (1.0 - rl_cfg.reward_baseline_beta) * reward
        )

        throughput[t] = rate
        reward_trace[t] = reward
        rate_delta_trace[t] = rate - zf_rate
        proxy_sinr_trace[t] = proxy_sinr
        zf_proxy_sinr_trace[t] = zf_proxy_sinr
        proxy_sinr_ratio_trace[t] = proxy_sinr / np.maximum(zf_proxy_sinr, rl_cfg.reward_sinr_eps)
        actual_sinr_ratio_trace[t] = actual_sinr / np.maximum(zf_actual_sinr, rl_cfg.reward_sinr_eps)
        advantage_trace[t] = advantage
        baseline_trace[t] = reward_baseline
        beat_zf_trace[t] = 1.0 if rate > zf_rate else 0.0
        beam_similarity_trace[t] = beam_similarity(beams, zf_baseline["precoders"][t])

        batch_s.append(s)
        batch_x.append(x_sample)
        batch_emp_mean.append(emp_mean_sample)
        batch_adv.append(advantage)
        batch_indices.append(t)

        if len(batch_s) >= rl_cfg.batch_size:
            flush_batch()

    flush_batch()
    print()

    return {
        "throughput": throughput,
        "reward": reward_trace,
        "rate_delta": rate_delta_trace,
        "proxy_sinr": proxy_sinr_trace,
        "zf_proxy_sinr": zf_proxy_sinr_trace,
        "proxy_sinr_ratio": proxy_sinr_ratio_trace,
        "actual_sinr_ratio": actual_sinr_ratio_trace,
        "advantage": advantage_trace,
        "reward_baseline": baseline_trace,
        "kappa": kappa_trace,
        "beat_zf": beat_zf_trace,
        "beam_similarity_to_zf": beam_similarity_trace,
        "grad_norm": grad_norm_trace,
        "loss": loss_trace,
        "signal": signal_trace,
        "leakage": leakage_trace,
        "fb_attempts": fb_attempts_trace,
        "fb_accept": fb_accept_trace,
        "best_of_n_score": best_of_n_score_trace,
        "best_of_n_selected": best_of_n_selected_trace,
        "empirical_mean_norm": empirical_mean_norm_trace,
        "centered_update_norm": centered_update_norm_trace,
        "wesn_states": wesn_states,
    }


def moving_average(trace: np.ndarray, window_len: int) -> np.ndarray:
    if window_len <= 1 or window_len > trace.size:
        return trace.copy()
    kernel = np.ones(window_len, dtype=np.float64) / window_len
    return np.convolve(trace, kernel, mode="valid")


def save_plots(
    zf_throughput: np.ndarray,
    random_vmf_throughput: np.ndarray,
    rl_results: dict[str, np.ndarray],
    output_dir: Path,
    window_len: int,
) -> None:
    rl_throughput = rl_results["throughput"]
    reward = rl_results["reward"]
    beat_zf = rl_results["beat_zf"]
    sim_to_zf = rl_results["beam_similarity_to_zf"].mean(axis=1)
    grad_norm = rl_results["grad_norm"]
    proxy_sinr_ratio = rl_results.get("proxy_sinr_ratio", None)
    actual_sinr_ratio = rl_results.get("actual_sinr_ratio", None)
    signal = rl_results.get("signal", None)
    leakage = rl_results.get("leakage", None)
    fb_attempts = rl_results.get("fb_attempts", None)
    fb_accept = rl_results.get("fb_accept", None)

    zf_avg = moving_average(zf_throughput, window_len)
    random_vmf_avg = moving_average(random_vmf_throughput, window_len)
    rl_avg = moving_average(rl_throughput, window_len)
    reward_avg = moving_average(reward, window_len)
    beat_zf_avg = moving_average(beat_zf, window_len)
    sim_avg = moving_average(sim_to_zf, window_len)
    grad_norm_avg = moving_average(grad_norm, window_len)
    proxy_sinr_ratio_avg = moving_average(proxy_sinr_ratio.mean(axis=1), window_len) if proxy_sinr_ratio is not None else None
    actual_sinr_ratio_avg = moving_average(actual_sinr_ratio.mean(axis=1), window_len) if actual_sinr_ratio is not None else None
    signal_avg = moving_average(signal.mean(axis=1), window_len) if signal is not None else None
    leakage_avg = moving_average(leakage.mean(axis=1), window_len) if leakage is not None else None
    fb_attempts_avg = moving_average(fb_attempts.mean(axis=1), window_len) if fb_attempts is not None else None
    fb_accept_avg = moving_average(fb_accept.mean(axis=1), window_len) if fb_accept is not None else None

    x_tput = np.arange(1, zf_avg.size + 1)
    x_reward = np.arange(1, reward_avg.size + 1)

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x_tput, zf_avg, lw=1.5, label="ZF baseline")
    ax1.plot(x_tput, random_vmf_avg, lw=1.5, label="Random vMF baseline")
    ax1.plot(x_tput, rl_avg, lw=1.5, label="WESN-FB PMI raw-SLNR RL")
    ax1.set_title("Throughput Across Time")
    ax1.set_xlabel("Slot index")
    ax1.set_ylabel("Sum-rate [bits/s/Hz]")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(output_dir / "throughput_across_time_zf_vs_esn_vmf_rl.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(x_reward, reward_avg, lw=1.5)
    ax2.set_title("WESN-FB Raw-SLNR RL Reward Across Time")
    ax2.set_xlabel("Slot index")
    ax2.set_ylabel("Reward vs ZF")
    ax2.grid(True, alpha=0.35)
    fig2.tight_layout()
    fig2.savefig(output_dir / "esn_vmf_rl_reward_across_time.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.plot(np.arange(1, beat_zf_avg.size + 1), beat_zf_avg, lw=1.5)
    ax3.set_title("Fraction of Slots Where WESN-FB Raw-SLNR RL Beats ZF")
    ax3.set_xlabel("Slot index")
    ax3.set_ylabel("Moving-average fraction")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.35)
    fig3.tight_layout()
    fig3.savefig(output_dir / "esn_vmf_rl_fraction_beats_zf.png", dpi=150)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(8, 4.5))
    ax4.plot(np.arange(1, sim_avg.size + 1), sim_avg, lw=1.5)
    ax4.set_title("Mean Beam Similarity to ZF")
    ax4.set_xlabel("Slot index")
    ax4.set_ylabel("Cosine similarity")
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(True, alpha=0.35)
    fig4.tight_layout()
    fig4.savefig(output_dir / "esn_vmf_rl_beam_similarity_to_zf.png", dpi=150)
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(8, 4.5))
    ax5.plot(np.arange(1, grad_norm_avg.size + 1), grad_norm_avg, lw=1.5)
    ax5.set_title("Policy Gradient Norm Across Time")
    ax5.set_xlabel("Slot index")
    ax5.set_ylabel("Gradient norm")
    ax5.grid(True, alpha=0.35)
    fig5.tight_layout()
    fig5.savefig(output_dir / "esn_fb_leakage_rl_grad_norm.png", dpi=150)
    plt.close(fig5)


    if proxy_sinr_ratio_avg is not None:
        fig_proxy, ax_proxy = plt.subplots(figsize=(8, 4.5))
        ax_proxy.plot(np.arange(1, proxy_sinr_ratio_avg.size + 1), proxy_sinr_ratio_avg, lw=1.5)
        ax_proxy.set_title("Mean Proxy-SINR Ratio to ZF")
        ax_proxy.set_xlabel("Slot index")
        ax_proxy.set_ylabel("Proxy-SINR ratio")
        ax_proxy.grid(True, alpha=0.35)
        fig_proxy.tight_layout()
        fig_proxy.savefig(output_dir / "esn_fb_proxy_sinr_ratio_to_zf.png", dpi=150)
        plt.close(fig_proxy)

    if actual_sinr_ratio_avg is not None:
        fig_actual, ax_actual = plt.subplots(figsize=(8, 4.5))
        ax_actual.plot(np.arange(1, actual_sinr_ratio_avg.size + 1), actual_sinr_ratio_avg, lw=1.5)
        ax_actual.set_title("Mean Actual SINR Ratio to ZF")
        ax_actual.set_xlabel("Slot index")
        ax_actual.set_ylabel("Actual SINR ratio")
        ax_actual.grid(True, alpha=0.35)
        fig_actual.tight_layout()
        fig_actual.savefig(output_dir / "esn_fb_actual_sinr_ratio_to_zf.png", dpi=150)
        plt.close(fig_actual)

    if signal_avg is not None:
        fig6, ax6 = plt.subplots(figsize=(8, 4.5))
        ax6.plot(np.arange(1, signal_avg.size + 1), signal_avg, lw=1.5)
        ax6.set_title("Mean Raw PMI Desired Signal of Accepted Samples")
        ax6.set_xlabel("Slot index")
        ax6.set_ylabel("Raw PMI desired signal")
        ax6.grid(True, alpha=0.35)
        fig6.tight_layout()
        fig6.savefig(output_dir / "esn_fb_normalized_signal.png", dpi=150)
        plt.close(fig6)

    if leakage_avg is not None:
        fig6b, ax6b = plt.subplots(figsize=(8, 4.5))
        ax6b.plot(np.arange(1, leakage_avg.size + 1), leakage_avg, lw=1.5)
        ax6b.set_title("Mean Raw PMI Leakage of Accepted Samples")
        ax6b.set_xlabel("Slot index")
        ax6b.set_ylabel("Raw PMI leakage")
        ax6b.grid(True, alpha=0.35)
        fig6b.tight_layout()
        fig6b.savefig(output_dir / "esn_fb_normalized_leakage.png", dpi=150)
        plt.close(fig6b)

    if fb_attempts_avg is not None:
        fig7, ax7 = plt.subplots(figsize=(8, 4.5))
        ax7.plot(np.arange(1, fb_attempts_avg.size + 1), fb_attempts_avg, lw=1.5)
        ax7.set_title("Fisher-Bingham Leakage Sampler Attempts")
        ax7.set_xlabel("Slot index")
        ax7.set_ylabel("Mean attempts per UE")
        ax7.grid(True, alpha=0.35)
        fig7.tight_layout()
        fig7.savefig(output_dir / "esn_fb_leakage_sampler_attempts.png", dpi=150)
        plt.close(fig7)

    if fb_accept_avg is not None:
        fig8, ax8 = plt.subplots(figsize=(8, 4.5))
        ax8.plot(np.arange(1, fb_accept_avg.size + 1), fb_accept_avg, lw=1.5)
        ax8.set_title("Fisher-Bingham Leakage Sampler Acceptance Rate")
        ax8.set_xlabel("Slot index")
        ax8.set_ylabel("Moving-average acceptance rate")
        ax8.set_ylim(0.0, 1.0)
        ax8.grid(True, alpha=0.35)
        fig8.tight_layout()
        fig8.savefig(output_dir / "esn_fb_leakage_sampler_acceptance.png", dpi=150)
        plt.close(fig8)



def sim_cache_metadata(cfg: SimConfig) -> dict[str, float | int]:
    """Metadata that determines whether cached channel-dependent baselines are valid."""
    return {
        "num_tx_antennas": cfg.num_tx_antennas,
        "num_users": cfg.num_users,
        "num_rx_antennas_per_user": cfg.num_rx_antennas_per_user,
        "streams_per_user": cfg.streams_per_user,
        "num_slots": cfg.num_slots,
        "snr_db": cfg.snr_db,
        "temporal_correlation": cfg.temporal_correlation,
        "seed": cfg.seed,
        "total_tx_power": cfg.total_tx_power,
    }


def _metadata_matches(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            actual = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def _write_metadata(path: Path, metadata: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_or_run_zf_baseline(
    cfg: SimConfig,
    channels: np.ndarray,
    cache_dir: Path,
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """Load ZF baseline from cache if compatible; otherwise compute and cache it.

    This avoids recomputing the deterministic ZF baseline when repeated runs use
    the same simulation configuration and channel seed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "zf_baseline_meta.json"
    metadata = sim_cache_metadata(cfg)

    paths = {
        "throughput": cache_dir / "zf_throughput_trace.npy",
        "sinr": cache_dir / "zf_sinr_trace.npy",
        "precoders": cache_dir / "zf_precoders_trace.npy",
        "pmi_features": cache_dir / "zf_pmi_features_trace.npy",
    }

    can_load = (
        not force_recompute
        and _metadata_matches(meta_path, metadata)
        and all(path.exists() for path in paths.values())
    )
    if can_load:
        print(f"Loading cached ZF baseline from {cache_dir}")
        return {name: np.load(path) for name, path in paths.items()}

    print("Cached ZF baseline not found or incompatible; recomputing.")
    zf_results = run_zf_baseline(cfg, channels)
    for name, path in paths.items():
        np.save(path, zf_results[name])
    _write_metadata(meta_path, metadata)
    return zf_results


def load_or_run_random_vmf_baseline(
    cfg: SimConfig,
    channels: np.ndarray,
    rng: np.random.Generator,
    fixed_kappa: float,
    cache_dir: Path,
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """Load random-vMF baseline from cache if compatible; otherwise compute and cache it.

    The random-vMF baseline depends on the simulation configuration, the channel
    seed, the fixed kappa, and the RNG seed used specifically for this baseline.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "random_vmf_baseline_meta.json"
    metadata = sim_cache_metadata(cfg) | {"fixed_kappa": fixed_kappa}
    throughput_path = cache_dir / "random_vmf_baseline_throughput_trace.npy"

    can_load = (
        not force_recompute
        and _metadata_matches(meta_path, metadata)
        and throughput_path.exists()
    )
    if can_load:
        print(f"Loading cached random vMF baseline from {cache_dir}")
        return {"throughput": np.load(throughput_path)}

    print("Cached random vMF baseline not found or incompatible; recomputing.")
    random_results = run_random_vmf_baseline(cfg, channels, rng, fixed_kappa=fixed_kappa)
    np.save(throughput_path, random_results["throughput"])
    _write_metadata(meta_path, metadata)
    return random_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple MU-MIMO ZF baseline + batched WESN PMI raw-SLNR RL")
    parser.add_argument("--num-slots", type=int, default=200000)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rl_precoder_design_per_UE"))
    parser.add_argument("--window-len", type=int, default=5000)
    parser.add_argument("--baseline-cache-dir", type=Path, default=None)
    parser.add_argument("--force-recompute-baselines", action="store_true")

    # WESN knobs. The reservoir update is the same as the vanilla ESN;
    # the readout is augmented with the current PMI input as a skip connection.
    # W_in and W_res are fixed; only W_out is learned.
    parser.add_argument("--reservoir-size", type=int, default=128)
    parser.add_argument("--spectral-radius", type=float, default=0.8)
    parser.add_argument("--input-scale", type=float, default=0.15)

    # RL/training knobs.
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-out", type=float, default=3e-2)
    parser.add_argument("--fixed-kappa", type=float, default=10.0)
    parser.add_argument("--reward-baseline-beta", type=float, default=0.99)
    parser.add_argument("--advantage-clip", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--init-scale-out", type=float, default=1e-2)
    parser.add_argument("--leakage-lambda", type=float, default=1.0, help="Deprecated/unused: SLNR-ratio denominator now uses normalized noise power.")
    parser.add_argument("--signal-gamma", type=float, default=1)
    parser.add_argument("--max-fb-resamples", type=int, default=16)
    parser.add_argument("--leakage-norm-eps", type=float, default=1e-12)
    parser.add_argument("--signal-norm-eps", type=float, default=1e-12)
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="actual_sinr_log_ratio",
        choices=[
            "throughput_delta",
            "normalized_throughput_delta",
            "actual_sinr_log_ratio",
            "sinr_proxy_log_ratio",
        ],
        help="Scalar reward used for REINFORCE updates.",
    )
    parser.add_argument("--reward-sinr-eps", type=float, default=1e-12)
    parser.add_argument("--best-of-n", type=int, default=1, help="Number of complete candidate precoders sampled per slot; execute/train on the best by reward score.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(
        num_slots=args.num_slots,
        snr_db=args.snr_db,
        temporal_correlation=args.rho,
        seed=args.seed,
    )
    rl_cfg = RLConfig(
        lr_out=args.lr_out,
        fixed_kappa=args.fixed_kappa,
        reward_baseline_beta=args.reward_baseline_beta,
        advantage_clip=args.advantage_clip,
        grad_clip_norm=args.grad_clip_norm,
        init_scale_out=args.init_scale_out,
        batch_size=args.batch_size,
        reservoir_size=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        input_scale=args.input_scale,
        leakage_lambda=args.leakage_lambda,
        signal_gamma=args.signal_gamma,
        max_fb_resamples=args.max_fb_resamples,
        leakage_norm_eps=args.leakage_norm_eps,
        signal_norm_eps=args.signal_norm_eps,
        reward_mode=args.reward_mode,
        reward_sinr_eps=args.reward_sinr_eps,
        best_of_n=args.best_of_n,
    )

    # Use separate RNG streams so loading cached baselines does not change the
    # stochastic trajectory of the RL policy. This keeps repeated runs comparable.
    rng_channels = np.random.default_rng(cfg.seed)
    rng_random_vmf = np.random.default_rng(cfg.seed + 10_000)
    rng_rl = np.random.default_rng(cfg.seed + 20_000)
    torch.manual_seed(cfg.seed)

    channels = simulate_channels(cfg, rng_channels)

    baseline_cache_dir = args.baseline_cache_dir
    if baseline_cache_dir is None:
        baseline_cache_dir = args.output_dir / "baseline_cache"

    zf_results = load_or_run_zf_baseline(
        cfg=cfg,
        channels=channels,
        cache_dir=baseline_cache_dir,
        force_recompute=args.force_recompute_baselines,
    )
    random_vmf_results = load_or_run_random_vmf_baseline(
        cfg=cfg,
        channels=channels,
        rng=rng_random_vmf,
        fixed_kappa=rl_cfg.fixed_kappa,
        cache_dir=baseline_cache_dir,
        force_recompute=args.force_recompute_baselines,
    )
    rl_results = run_esn_policy_rl(cfg, rl_cfg, channels, zf_results, rng_rl)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "zf_throughput_trace.npy", zf_results["throughput"])
    np.save(args.output_dir / "random_vmf_baseline_throughput_trace.npy", random_vmf_results["throughput"])
    np.save(args.output_dir / "esn_vmf_rl_throughput_trace.npy", rl_results["throughput"])
    np.save(args.output_dir / "esn_vmf_rl_reward_trace.npy", rl_results["reward"])
    np.save(args.output_dir / "esn_vmf_rl_rate_delta_trace.npy", rl_results["rate_delta"])
    np.save(args.output_dir / "esn_fb_proxy_sinr_trace.npy", rl_results["proxy_sinr"])
    np.save(args.output_dir / "esn_fb_zf_proxy_sinr_trace.npy", rl_results["zf_proxy_sinr"])
    np.save(args.output_dir / "esn_fb_proxy_sinr_ratio_trace.npy", rl_results["proxy_sinr_ratio"])
    np.save(args.output_dir / "esn_fb_actual_sinr_ratio_trace.npy", rl_results["actual_sinr_ratio"])
    np.save(args.output_dir / "esn_vmf_rl_advantage_trace.npy", rl_results["advantage"])
    np.save(args.output_dir / "esn_vmf_rl_reward_baseline_trace.npy", rl_results["reward_baseline"])
    np.save(args.output_dir / "esn_vmf_rl_kappa_trace.npy", rl_results["kappa"])
    np.save(args.output_dir / "esn_vmf_rl_fraction_beats_zf_trace.npy", rl_results["beat_zf"])
    np.save(args.output_dir / "esn_vmf_rl_beam_similarity_to_zf_trace.npy", rl_results["beam_similarity_to_zf"])
    np.save(args.output_dir / "esn_vmf_rl_grad_norm_trace.npy", rl_results["grad_norm"])
    np.save(args.output_dir / "esn_vmf_rl_loss_trace.npy", rl_results["loss"])
    np.save(args.output_dir / "esn_fb_signal_trace.npy", rl_results["signal"])
    np.save(args.output_dir / "esn_fb_leakage_trace.npy", rl_results["leakage"])
    np.save(args.output_dir / "esn_fb_sampler_attempts_trace.npy", rl_results["fb_attempts"])
    np.save(args.output_dir / "esn_fb_sampler_accept_trace.npy", rl_results["fb_accept"])
    np.save(args.output_dir / "esn_fb_best_of_n_score_trace.npy", rl_results["best_of_n_score"])
    np.save(args.output_dir / "esn_fb_best_of_n_selected_trace.npy", rl_results["best_of_n_selected"])
    np.save(args.output_dir / "esn_fb_empirical_mean_norm_trace.npy", rl_results["empirical_mean_norm"])
    np.save(args.output_dir / "esn_fb_centered_update_norm_trace.npy", rl_results["centered_update_norm"])
    np.save(args.output_dir / "wesn_states_trace.npy", rl_results["wesn_states"])

    save_plots(
        zf_throughput=zf_results["throughput"],
        random_vmf_throughput=random_vmf_results["throughput"],
        rl_results=rl_results,
        output_dir=args.output_dir,
        window_len=args.window_len,
    )

    print("Simple WESN-FB PMI raw-SLNR RL precoder design run finished.")
    print(f"ZF average throughput         : {zf_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Random vMF baseline throughput: {random_vmf_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"WESN-FB PMI raw-SLNR RL throughput  : {rl_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Reward mode                   : {rl_cfg.reward_mode}")
    print(f"WESN-FB PMI raw-SLNR RL reward      : {rl_results['reward'].mean():.4f}")
    print(f"WESN-FB throughput delta       : {rl_results['rate_delta'].mean():.4f}")
    print(f"Mean proxy-SINR ratio to ZF   : {rl_results['proxy_sinr_ratio'].mean():.4f}")
    print(f"Mean actual SINR ratio to ZF  : {rl_results['actual_sinr_ratio'].mean():.4f}")
    print(f"RL beats ZF fraction          : {rl_results['beat_zf'].mean():.4f}")
    print(f"Mean beam similarity to ZF    : {rl_results['beam_similarity_to_zf'].mean():.4f}")
    print(f"Mean grad norm                : {rl_results['grad_norm'].mean():.4f}")
    print(f"Mean raw PMI desired signal: {rl_results['signal'].mean():.4f}")
    print(f"Mean raw PMI leakage       : {rl_results['leakage'].mean():.4f}")
    print(f"Empirical mean correction  : yes (centered x - E_shaped[x])")
    print(f"Mean empirical mean norm   : {rl_results['empirical_mean_norm'].mean():.4f}")
    print(f"Mean centered update norm  : {rl_results['centered_update_norm'].mean():.4f}")
    print(f"SLNR score uses raw PMI-only actual SLNR: yes")
    print(f"FB sampler acceptance rate    : {rl_results['fb_accept'].mean():.4f}")
    print(f"FB sampler avg attempts       : {rl_results['fb_attempts'].mean():.4f}")
    print(f"Best-of-N candidates          : {rl_cfg.best_of_n}")
    print(f"Mean selected candidate index : {rl_results['best_of_n_selected'].mean():.4f}")


if __name__ == "__main__":
    main()
