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
    reward_mode: str = "sinr_proxy_log_ratio"
    reward_sinr_eps: float = 1e-12
    best_of_n: int = 1
    global_proxy_eta: float = 1.0
    global_gibbs_sweeps: int = 1
    global_conditional_candidates: int = 8
    global_softmax_temperature: float = 0.0


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

    The default is sinr_proxy_log_ratio because it gives a balanced per-user
    SINR-like signal and penalizes cases where one user's SINR collapses.
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




def phase_invariant_alignment_score(x: np.ndarray, mu: np.ndarray) -> float:
    """Return |m^H v|^2 using real representations x=[Re{v}; Im{v}].

    This is invariant to an arbitrary complex scalar phase applied to either
    the sampled beam v or the mean beam m. Both x and mu are assumed to be
    real representations of unit-norm complex vectors.
    """
    nt = x.size // 2
    x_re, x_im = x[:nt], x[nt:]
    m_re, m_im = mu[:nt], mu[nt:]
    inner_re = float(m_re @ x_re + m_im @ x_im)
    inner_im = float(m_re @ x_im - m_im @ x_re)
    return inner_re * inner_re + inner_im * inner_im




def apply_random_global_phase_real(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply v <- exp(j phi) v to real representation x=[Re{v}; Im{v}]."""
    nt = x.size // 2
    phi = rng.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(phi), np.sin(phi)
    re = x[:nt]
    im = x[nt:]
    return np.concatenate([c * re - s * im, s * re + c * im]).astype(np.float64)


def sample_phase_randomized_vmf(mu: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Sample near the complex line spanned by mu, then randomize global phase.

    Standard real-vMF around a fixed real vector is not phase invariant. This
    helper uses it only as an efficient proposal near the ESN-predicted line and
    then applies a uniform global phase rotation, so all phase-equivalent beams
    are treated identically at execution time.
    """
    x = sample_vmf(mu, kappa, rng)
    return apply_random_global_phase_real(x, rng)

def real_to_complex_unit(x: np.ndarray) -> np.ndarray:
    """Convert real representation x=[Re{v}; Im{v}] to a unit-norm complex vector."""
    nt = x.size // 2
    v = x[:nt] + 1j * x[nt:]
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        v = np.ones(nt, dtype=np.complex128) / np.sqrt(nt)
    else:
        v = v / nrm
    return v


def complex_unit_to_real(v: np.ndarray) -> np.ndarray:
    """Convert a complex vector to normalized real representation [Re{v}; Im{v}]."""
    v = v.astype(np.complex128, copy=False)
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        v = np.ones_like(v) / np.sqrt(v.size)
    else:
        v = v / nrm
    return np.concatenate([np.real(v), np.imag(v)]).astype(np.float64)


def pmi_dominant_vectors_from_channels(user_channels: np.ndarray) -> np.ndarray:
    """Return one dominant right singular vector q_k per UE.

    This function is used only to emulate PMI/right-singular-vector feedback in
    the toy testbench.  The global policy shaping below uses these q_k vectors,
    not the full channel Gram matrices H_k^H H_k.
    """
    q_list = []
    for k in range(user_channels.shape[0]):
        _, _, vh = np.linalg.svd(user_channels[k], full_matrices=True)
        q = vh.conj().T[:, 0]
        q_list.append(q / max(np.linalg.norm(q), 1e-12))
    return np.stack(q_list, axis=0)


def pmi_proxy_sinr_from_unit_beams(
    q_vectors: np.ndarray,
    v_unit: np.ndarray,
    snr_linear: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """PMI-only equal-power SINR proxy using right singular vectors.

    q_vectors[k] is the PMI/right singular vector reported by UE k.
    v_unit[:, j] is the unit-norm beam direction for stream/user j.

        SINR_hat_k = |q_k^H v_k|^2 /
                     (sum_{j != k} |q_k^H v_j|^2 + K/SNR_linear).

    This uses only q_k directions and the SNR, not the full channel matrices.
    """
    k_users = q_vectors.shape[0]
    noise_term = k_users / max(snr_linear, eps)
    out = np.zeros(k_users, dtype=np.float64)
    for k in range(k_users):
        qk = q_vectors[k]
        desired = float(np.abs(np.vdot(qk, v_unit[:, k])) ** 2)
        interf = 0.0
        for j in range(k_users):
            if j != k:
                interf += float(np.abs(np.vdot(qk, v_unit[:, j])) ** 2)
        out[k] = desired / (interf + noise_term + eps)
    return out


def pmi_proxy_sum_rate_from_unit_beams(
    q_vectors: np.ndarray,
    v_unit: np.ndarray,
    snr_linear: float,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Return sum_k log(1 + PMI-SINR_k) and per-user PMI-SINR."""
    sinr = pmi_proxy_sinr_from_unit_beams(q_vectors, v_unit, snr_linear, eps=eps)
    return float(np.sum(np.log1p(sinr))), sinr


def pmi_proxy_sum_rate_from_x(
    q_vectors: np.ndarray,
    x_sample: np.ndarray,
    snr_linear: float,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """PMI proxy sum-rate for x_sample with shape [K, 2*N_t]."""
    k_users = x_sample.shape[0]
    nt = x_sample.shape[1] // 2
    v_unit = np.zeros((nt, k_users), dtype=np.complex128)
    for k in range(k_users):
        v_unit[:, k] = real_to_complex_unit(x_sample[k])
    return pmi_proxy_sum_rate_from_unit_beams(q_vectors, v_unit, snr_linear, eps=eps)


def global_joint_pmi_proxy_score(
    x_sample: np.ndarray,
    mu: np.ndarray,
    q_vectors: np.ndarray,
    kappa: float,
    eta: float,
    snr_linear: float,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Unnormalized global product-sphere score.

        S(X) = kappa * sum_k |m_k^H v_k|^2
               + eta * sum_k log(1 + SINR_hat_k(X)).

    The constraints ||v_k||=1 are enforced by the representation/sampler.
    """
    align = np.array(
        [phase_invariant_alignment_score(x_sample[k], mu[k]) for k in range(x_sample.shape[0])],
        dtype=np.float64,
    )
    pmi_sum_rate, pmi_sinr = pmi_proxy_sum_rate_from_x(
        q_vectors=q_vectors,
        x_sample=x_sample,
        snr_linear=snr_linear,
        eps=eps,
    )
    score = float(kappa * np.sum(align) + eta * pmi_sum_rate)
    return score, align, pmi_sum_rate, pmi_sinr


def _choose_index_from_scores(scores: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Choose candidate by argmax if temperature <= 0, otherwise softmax sample."""
    if temperature <= 0.0:
        return int(np.argmax(scores))
    scaled = scores / max(temperature, 1e-12)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / np.sum(probs)
    return int(rng.choice(np.arange(scores.size), p=probs))


def sample_global_joint_pmi_proxy_product_sphere(
    mu: np.ndarray,
    q_vectors: np.ndarray,
    kappa: float,
    eta: float,
    snr_linear: float,
    rng: np.random.Generator,
    gibbs_sweeps: int,
    conditional_candidates: int,
    softmax_temperature: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Approximate sample from a global joint distribution on a product of spheres.

    Target score:

        S(V) = kappa * sum_k |m_k^H v_k|^2
               + eta * sum_k log(1 + SINR_hat_k(V)),

    where SINR_hat is computed only from PMI/right singular vectors q_k:

        SINR_hat_k = |q_k^H v_k|^2 /
                     (sum_{j != k} |q_k^H v_j|^2 + K/SNR).

    The sampler uses Gibbs-style conditional proposal/selection.  Each column
    remains unit norm, so equal power and total power are preserved after
    scaling by sqrt(P_tot/K).
    """
    k_users, d = mu.shape
    # Initialize each column near its learned complex line.
    x = np.zeros((k_users, d), dtype=np.float64)
    for k in range(k_users):
        x[k] = sample_phase_randomized_vmf(mu[k], kappa, rng)

    sweeps = max(1, int(gibbs_sweeps))
    m_cands = max(1, int(conditional_candidates))

    for _ in range(sweeps):
        for k in range(k_users):
            cand_x = []
            cand_scores = np.zeros(m_cands, dtype=np.float64)
            for m in range(m_cands):
                x_prop = x.copy()
                x_prop[k] = sample_phase_randomized_vmf(mu[k], kappa, rng)
                cand_x.append(x_prop[k])
                cand_scores[m], _, _, _ = global_joint_pmi_proxy_score(
                    x_sample=x_prop,
                    mu=mu,
                    q_vectors=q_vectors,
                    kappa=kappa,
                    eta=eta,
                    snr_linear=snr_linear,
                    eps=eps,
                )
            chosen = _choose_index_from_scores(cand_scores, softmax_temperature, rng)
            x[k] = cand_x[chosen]

    final_score, align, pmi_sum_rate, pmi_sinr = global_joint_pmi_proxy_score(
        x_sample=x,
        mu=mu,
        q_vectors=q_vectors,
        kappa=kappa,
        eta=eta,
        snr_linear=snr_linear,
        eps=eps,
    )
    return x, align, pmi_sum_rate, pmi_sinr, final_score

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


def compute_esn_states(pmi_features: np.ndarray, rl_cfg: RLConfig, rng: np.random.Generator) -> np.ndarray:
    """Compute fixed ESN states from all-UE PMI features.

    The ESN is fixed. Only W_out is trained by the RL algorithm.
    State feature used by W_out is [Re{z_t}, Im{z_t}, 1].
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
    states = np.zeros((num_slots, 2 * nz + 1), dtype=np.float64)
    for t in range(num_slots):
        y = pmi_features[t]
        z = split_tanh_np(w_in @ y + w_res @ z)
        states[t, :-1] = np.concatenate([np.real(z), np.imag(z)])
        states[t, -1] = 1.0
    return states


def phase_invariant_log_score_fixed_kappa_torch(
    x: torch.Tensor, mu: torch.Tensor, fixed_kappa: float
) -> torch.Tensor:
    """Phase-invariant Bingham-style alignment score.

    Returns fixed_kappa * |m^H v|^2, where x=[Re{v}; Im{v}]
    and mu=[Re{m}; Im{m}]. This replaces the old real-vMF score
    fixed_kappa * mu^T x, which was not invariant to per-stream phase.

    This is used as an unnormalized policy log-score for REINFORCE.
    As with the earlier Fisher-Bingham sampler, the normalizer is omitted
    as a practical approximation.
    """
    d = x.shape[-1]
    nt = d // 2
    x_re, x_im = x[..., :nt], x[..., nt:]
    m_re, m_im = mu[..., :nt], mu[..., nt:]
    inner_re = torch.sum(m_re * x_re + m_im * x_im, dim=-1)
    inner_im = torch.sum(m_re * x_im - m_im * x_re, dim=-1)
    alignment = inner_re.square() + inner_im.square()
    return fixed_kappa * alignment


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

    # Shared ESN state generated from the PMI feedback of all UEs.
    esn_states = compute_esn_states(zf_baseline["pmi_features"], rl_cfg, rng)
    state_dim = esn_states.shape[1]

    # W_out maps shared ESN state to all UE vMF mean logits, shape [K, D, N_z_aug].
    # This is the only trainable policy parameter in this fixed-reservoir ESN setup.
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
    alignment_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    pmi_shaping_sinr_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    pmi_shaping_sum_rate_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    conditional_candidates_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    best_of_n_score_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    best_of_n_selected_trace = np.zeros(cfg.num_slots, dtype=np.float64)

    reward_baseline: float | None = None
    batch_s: list[np.ndarray] = []
    batch_x: list[np.ndarray] = []
    batch_adv: list[float] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        nonlocal batch_s, batch_x, batch_adv, batch_indices
        if not batch_s:
            return

        s_batch = torch.tensor(np.stack(batch_s, axis=0), dtype=torch.float64)
        x_batch = torch.tensor(np.stack(batch_x, axis=0), dtype=torch.float64)
        adv_batch = torch.tensor(np.array(batch_adv), dtype=torch.float64)

        # Recompute log probabilities under the current W_out for REINFORCE.
        # logits/mu shape: [B, K, D]
        logits = torch.einsum("kdn,bn->bkd", w_out, s_batch)
        mu = logits / torch.clamp(torch.linalg.norm(logits, dim=-1, keepdim=True), min=1e-12)

        per_user_log_prob = phase_invariant_log_score_fixed_kappa_torch(x_batch, mu, rl_cfg.fixed_kappa)
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
        batch_adv = []
        batch_indices = []

    for t in range(cfg.num_slots):
        print(f"ESN-RL Slot {t + 1} / {cfg.num_slots}", end="\r")
        s = esn_states[t]

        # All users' vMF means are produced from the same shared ESN state.
        s_t = torch.tensor(s, dtype=torch.float64)
        with torch.no_grad():
            logits = torch.einsum("kdn,n->kd", w_out, s_t)
            mu_t = logits / torch.clamp(torch.linalg.norm(logits, dim=-1, keepdim=True), min=1e-12)
            mu_np = mu_t.detach().cpu().numpy()

        # Emulate the PMI/right-singular-vector feedback available to the policy.
        # The global joint sampler below uses only these q-vectors for distribution
        # shaping, not the full channel Gram matrices H_k^H H_k.
        q_vectors = pmi_dominant_vectors_from_channels(channels[t])
        snr_linear = 10.0 ** (cfg.snr_db / 10.0)

        zf_rate = float(zf_baseline["throughput"][t])
        zf_actual_sinr = zf_baseline["sinr"][t]
        zf_proxy_sinr = compute_slot_sinr_proxy(
            channels[t], zf_baseline["precoders"][t], noise_power
        )

        # Best-of-N over samples from one global joint product-sphere distribution.
        # Each candidate is generated as a complete K-column matrix using the
        # PMI proxy sum-rate coupled sampler.  Set --best-of-n 1 to disable the
        # outer selection and use one global sample per slot.
        best_score = -np.inf
        best_candidate_index = 0
        best_beams = None
        best_x_sample = None
        best_alignment = None
        best_pmi_sum_rate = None
        best_pmi_sinr = None
        best_joint_score = None
        best_rate = None
        best_actual_sinr = None
        best_proxy_sinr = None
        best_reward = None

        for cand_idx in range(max(1, rl_cfg.best_of_n)):
            cand_x_sample, cand_alignment, cand_pmi_sum_rate, cand_pmi_sinr, cand_joint_score = (
                sample_global_joint_pmi_proxy_product_sphere(
                    mu=mu_np,
                    q_vectors=q_vectors,
                    kappa=rl_cfg.fixed_kappa,
                    eta=rl_cfg.global_proxy_eta,
                    snr_linear=snr_linear,
                    rng=rng,
                    gibbs_sweeps=rl_cfg.global_gibbs_sweeps,
                    conditional_candidates=rl_cfg.global_conditional_candidates,
                    softmax_temperature=rl_cfg.global_softmax_temperature,
                    eps=rl_cfg.reward_sinr_eps,
                )
            )

            cand_beams = np.zeros((cfg.num_tx_antennas, k), dtype=np.complex128)
            for ku in range(k):
                cand_beams[:, ku] = real_to_complex_beam(cand_x_sample[ku], cfg.total_tx_power, k)

            cand_rate, cand_actual_sinr = compute_slot_sum_rate(
                channels[t], cand_beams, noise_power
            )
            cand_proxy_sinr = compute_slot_sinr_proxy(
                channels[t], cand_beams, noise_power
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

            # Candidate selection is still based on the RL reward used for the
            # environment update, while candidate generation is globally shaped
            # by PMI proxy sum-rate.
            if cand_reward > best_score:
                best_score = cand_reward
                best_candidate_index = cand_idx
                best_beams = cand_beams
                best_x_sample = cand_x_sample
                best_alignment = cand_alignment
                best_pmi_sum_rate = cand_pmi_sum_rate
                best_pmi_sinr = cand_pmi_sinr
                best_joint_score = cand_joint_score
                best_rate = cand_rate
                best_actual_sinr = cand_actual_sinr
                best_proxy_sinr = cand_proxy_sinr
                best_reward = cand_reward

        assert best_beams is not None
        beams = best_beams
        x_sample = best_x_sample
        alignment_trace[t] = best_alignment
        pmi_shaping_sinr_trace[t] = best_pmi_sinr
        pmi_shaping_sum_rate_trace[t] = float(best_pmi_sum_rate)
        conditional_candidates_trace[t] = rl_cfg.global_conditional_candidates
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
        "phase_alignment": alignment_trace,
        "advantage": advantage_trace,
        "reward_baseline": baseline_trace,
        "kappa": kappa_trace,
        "beat_zf": beat_zf_trace,
        "beam_similarity_to_zf": beam_similarity_trace,
        "grad_norm": grad_norm_trace,
        "loss": loss_trace,
        "pmi_shaping_sinr": pmi_shaping_sinr_trace,
        "pmi_shaping_sum_rate": pmi_shaping_sum_rate_trace,
        "conditional_candidates": conditional_candidates_trace,
        "best_of_n_score": best_of_n_score_trace,
        "best_of_n_selected": best_of_n_selected_trace,
        "esn_states": esn_states,
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
    phase_alignment = rl_results.get("phase_alignment", None)
    pmi_shaping_sinr = rl_results.get("pmi_shaping_sinr", None)
    pmi_shaping_sum_rate = rl_results.get("pmi_shaping_sum_rate", None)
    conditional_candidates = rl_results.get("conditional_candidates", None)

    zf_avg = moving_average(zf_throughput, window_len)
    random_vmf_avg = moving_average(random_vmf_throughput, window_len)
    rl_avg = moving_average(rl_throughput, window_len)
    reward_avg = moving_average(reward, window_len)
    beat_zf_avg = moving_average(beat_zf, window_len)
    sim_avg = moving_average(sim_to_zf, window_len)
    grad_norm_avg = moving_average(grad_norm, window_len)
    proxy_sinr_ratio_avg = moving_average(proxy_sinr_ratio.mean(axis=1), window_len) if proxy_sinr_ratio is not None else None
    actual_sinr_ratio_avg = moving_average(actual_sinr_ratio.mean(axis=1), window_len) if actual_sinr_ratio is not None else None
    phase_alignment_avg = moving_average(phase_alignment.mean(axis=1), window_len) if phase_alignment is not None else None
    pmi_shaping_sinr_avg = moving_average(pmi_shaping_sinr.mean(axis=1), window_len) if pmi_shaping_sinr is not None else None
    pmi_shaping_sum_rate_avg = moving_average(pmi_shaping_sum_rate, window_len) if pmi_shaping_sum_rate is not None else None
    conditional_candidates_avg = moving_average(conditional_candidates, window_len) if conditional_candidates is not None else None

    x_tput = np.arange(1, zf_avg.size + 1)
    x_reward = np.arange(1, reward_avg.size + 1)

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x_tput, zf_avg, lw=1.5, label="ZF baseline")
    ax1.plot(x_tput, random_vmf_avg, lw=1.5, label="Random vMF baseline")
    ax1.plot(x_tput, rl_avg, lw=1.5, label="ESN global joint PMI-proxy RL")
    ax1.set_title("Throughput Across Time")
    ax1.set_xlabel("Slot index")
    ax1.set_ylabel("Sum-rate [bits/s/Hz]")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(output_dir / "throughput_across_time_zf_vs_esn_global_rl.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(x_reward, reward_avg, lw=1.5)
    ax2.set_title("ESN Global Joint PMI-Proxy RL Reward Across Time")
    ax2.set_xlabel("Slot index")
    ax2.set_ylabel("Reward vs ZF")
    ax2.grid(True, alpha=0.35)
    fig2.tight_layout()
    fig2.savefig(output_dir / "esn_global_rl_reward_across_time.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.plot(np.arange(1, beat_zf_avg.size + 1), beat_zf_avg, lw=1.5)
    ax3.set_title("Fraction of Slots Where ESN Global Joint PMI-Proxy RL Beats ZF")
    ax3.set_xlabel("Slot index")
    ax3.set_ylabel("Moving-average fraction")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.35)
    fig3.tight_layout()
    fig3.savefig(output_dir / "esn_global_rl_fraction_beats_zf.png", dpi=150)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(8, 4.5))
    ax4.plot(np.arange(1, sim_avg.size + 1), sim_avg, lw=1.5)
    ax4.set_title("Mean Beam Similarity to ZF")
    ax4.set_xlabel("Slot index")
    ax4.set_ylabel("Cosine similarity")
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(True, alpha=0.35)
    fig4.tight_layout()
    fig4.savefig(output_dir / "esn_global_rl_beam_similarity_to_zf.png", dpi=150)
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(8, 4.5))
    ax5.plot(np.arange(1, grad_norm_avg.size + 1), grad_norm_avg, lw=1.5)
    ax5.set_title("Policy Gradient Norm Across Time")
    ax5.set_xlabel("Slot index")
    ax5.set_ylabel("Gradient norm")
    ax5.grid(True, alpha=0.35)
    fig5.tight_layout()
    fig5.savefig(output_dir / "esn_global_rl_grad_norm.png", dpi=150)
    plt.close(fig5)


    if proxy_sinr_ratio_avg is not None:
        fig_proxy, ax_proxy = plt.subplots(figsize=(8, 4.5))
        ax_proxy.plot(np.arange(1, proxy_sinr_ratio_avg.size + 1), proxy_sinr_ratio_avg, lw=1.5)
        ax_proxy.set_title("Mean Proxy-SINR Ratio to ZF")
        ax_proxy.set_xlabel("Slot index")
        ax_proxy.set_ylabel("Proxy-SINR ratio")
        ax_proxy.grid(True, alpha=0.35)
        fig_proxy.tight_layout()
        fig_proxy.savefig(output_dir / "esn_global_proxy_sinr_ratio_to_zf.png", dpi=150)
        plt.close(fig_proxy)

    if actual_sinr_ratio_avg is not None:
        fig_actual, ax_actual = plt.subplots(figsize=(8, 4.5))
        ax_actual.plot(np.arange(1, actual_sinr_ratio_avg.size + 1), actual_sinr_ratio_avg, lw=1.5)
        ax_actual.set_title("Mean Actual SINR Ratio to ZF")
        ax_actual.set_xlabel("Slot index")
        ax_actual.set_ylabel("Actual SINR ratio")
        ax_actual.grid(True, alpha=0.35)
        fig_actual.tight_layout()
        fig_actual.savefig(output_dir / "esn_global_actual_sinr_ratio_to_zf.png", dpi=150)
        plt.close(fig_actual)

    if phase_alignment_avg is not None:
        fig_align, ax_align = plt.subplots(figsize=(8, 4.5))
        ax_align.plot(np.arange(1, phase_alignment_avg.size + 1), phase_alignment_avg, lw=1.5)
        ax_align.set_title("Mean Phase-Invariant Alignment of Accepted Samples")
        ax_align.set_xlabel("Slot index")
        ax_align.set_ylabel(r"$|m^H v|^2$")
        ax_align.set_ylim(0.0, 1.0)
        ax_align.grid(True, alpha=0.35)
        fig_align.tight_layout()
        fig_align.savefig(output_dir / "esn_global_phase_invariant_alignment.png", dpi=150)
        plt.close(fig_align)

    if pmi_shaping_sinr_avg is not None:
        fig6, ax6 = plt.subplots(figsize=(8, 4.5))
        ax6.plot(np.arange(1, pmi_shaping_sinr_avg.size + 1), pmi_shaping_sinr_avg, lw=1.5)
        ax6.set_title("Mean PMI-Proxy SINR Used by Global Sampler")
        ax6.set_xlabel("Slot index")
        ax6.set_ylabel("PMI-proxy SINR")
        ax6.grid(True, alpha=0.35)
        fig6.tight_layout()
        fig6.savefig(output_dir / "esn_global_pmi_proxy_sinr.png", dpi=150)
        plt.close(fig6)

    if pmi_shaping_sum_rate_avg is not None:
        fig7, ax7 = plt.subplots(figsize=(8, 4.5))
        ax7.plot(np.arange(1, pmi_shaping_sum_rate_avg.size + 1), pmi_shaping_sum_rate_avg, lw=1.5)
        ax7.set_title("PMI-Proxy Sum-Rate Used by Global Sampler")
        ax7.set_xlabel("Slot index")
        ax7.set_ylabel("PMI-proxy sum-rate")
        ax7.grid(True, alpha=0.35)
        fig7.tight_layout()
        fig7.savefig(output_dir / "esn_global_pmi_proxy_sum_rate.png", dpi=150)
        plt.close(fig7)

    if conditional_candidates_avg is not None:
        fig8, ax8 = plt.subplots(figsize=(8, 4.5))
        ax8.plot(np.arange(1, conditional_candidates_avg.size + 1), conditional_candidates_avg, lw=1.5)
        ax8.set_title("Global Conditional Candidates per Column")
        ax8.set_xlabel("Slot index")
        ax8.set_ylabel("Conditional candidates")
        ax8.grid(True, alpha=0.35)
        fig8.tight_layout()
        fig8.savefig(output_dir / "esn_global_conditional_candidates.png", dpi=150)
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
    parser = argparse.ArgumentParser(description="Simple MU-MIMO ZF baseline + batched ESN global joint PMI-proxy product-sphere RL")
    parser.add_argument("--num-slots", type=int, default=200000)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/simple_esn_global_joint_pmi_proxy_rl_precoder_design"))
    parser.add_argument("--window-len", type=int, default=1000)
    parser.add_argument("--baseline-cache-dir", type=Path, default=None)
    parser.add_argument("--force-recompute-baselines", action="store_true")

    # ESN knobs. W_in and W_res are fixed; only W_out is learned.
    parser.add_argument("--reservoir-size", type=int, default=128)
    parser.add_argument("--spectral-radius", type=float, default=0.8)
    parser.add_argument("--input-scale", type=float, default=0.15)

    # RL/training knobs.
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr-out", type=float, default=3e-2)
    parser.add_argument("--fixed-kappa", type=float, default=10.0)
    parser.add_argument("--reward-baseline-beta", type=float, default=0.99)
    parser.add_argument("--advantage-clip", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--init-scale-out", type=float, default=1e-2)
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="sinr_proxy_log_ratio",
        choices=[
            "throughput_delta",
            "normalized_throughput_delta",
            "actual_sinr_log_ratio",
            "sinr_proxy_log_ratio",
        ],
        help="Scalar reward used for REINFORCE updates.",
    )
    parser.add_argument("--reward-sinr-eps", type=float, default=1e-12)
    parser.add_argument("--best-of-n", type=int, default=1, help="Number of complete global joint candidate precoders sampled per slot; execute/train on the best by reward score.")
    parser.add_argument("--global-proxy-eta", type=float, default=1.0, help="Weight eta for PMI proxy sum-rate in the global joint product-sphere sampler.")
    parser.add_argument("--global-gibbs-sweeps", type=int, default=1, help="Number of Gibbs-style sweeps used to construct one global joint sample.")
    parser.add_argument("--global-conditional-candidates", type=int, default=8, help="Number of candidate directions tested for each column update in the global joint sampler.")
    parser.add_argument("--global-softmax-temperature", type=float, default=0.0, help="If >0, sample conditional candidates by softmax(score/temp); if 0, use argmax.")
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
        reward_mode=args.reward_mode,
        reward_sinr_eps=args.reward_sinr_eps,
        best_of_n=args.best_of_n,
        global_proxy_eta=args.global_proxy_eta,
        global_gibbs_sweeps=args.global_gibbs_sweeps,
        global_conditional_candidates=args.global_conditional_candidates,
        global_softmax_temperature=args.global_softmax_temperature,
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
    np.save(args.output_dir / "esn_global_rl_throughput_trace.npy", rl_results["throughput"])
    np.save(args.output_dir / "esn_global_rl_reward_trace.npy", rl_results["reward"])
    np.save(args.output_dir / "esn_global_rl_rate_delta_trace.npy", rl_results["rate_delta"])
    np.save(args.output_dir / "esn_global_proxy_sinr_trace.npy", rl_results["proxy_sinr"])
    np.save(args.output_dir / "esn_global_zf_proxy_sinr_trace.npy", rl_results["zf_proxy_sinr"])
    np.save(args.output_dir / "esn_global_proxy_sinr_ratio_trace.npy", rl_results["proxy_sinr_ratio"])
    np.save(args.output_dir / "esn_global_actual_sinr_ratio_trace.npy", rl_results["actual_sinr_ratio"])
    np.save(args.output_dir / "esn_global_phase_invariant_alignment_trace.npy", rl_results["phase_alignment"])
    np.save(args.output_dir / "esn_global_rl_advantage_trace.npy", rl_results["advantage"])
    np.save(args.output_dir / "esn_global_rl_reward_baseline_trace.npy", rl_results["reward_baseline"])
    np.save(args.output_dir / "esn_global_rl_kappa_trace.npy", rl_results["kappa"])
    np.save(args.output_dir / "esn_global_rl_fraction_beats_zf_trace.npy", rl_results["beat_zf"])
    np.save(args.output_dir / "esn_global_rl_beam_similarity_to_zf_trace.npy", rl_results["beam_similarity_to_zf"])
    np.save(args.output_dir / "esn_global_rl_grad_norm_trace.npy", rl_results["grad_norm"])
    np.save(args.output_dir / "esn_global_rl_loss_trace.npy", rl_results["loss"])
    np.save(args.output_dir / "esn_global_pmi_shaping_sinr_trace.npy", rl_results["pmi_shaping_sinr"])
    np.save(args.output_dir / "esn_global_pmi_shaping_sum_rate_trace.npy", rl_results["pmi_shaping_sum_rate"])
    np.save(args.output_dir / "esn_global_conditional_candidates_trace.npy", rl_results["conditional_candidates"])
    np.save(args.output_dir / "esn_global_best_of_n_score_trace.npy", rl_results["best_of_n_score"])
    np.save(args.output_dir / "esn_global_best_of_n_selected_trace.npy", rl_results["best_of_n_selected"])
    np.save(args.output_dir / "esn_states_trace.npy", rl_results["esn_states"])

    save_plots(
        zf_throughput=zf_results["throughput"],
        random_vmf_throughput=random_vmf_results["throughput"],
        rl_results=rl_results,
        output_dir=args.output_dir,
        window_len=args.window_len,
    )

    print("Simple ESN global joint PMI-proxy product-sphere RL precoder design run finished.")
    print(f"ZF average throughput         : {zf_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Random vMF baseline throughput: {random_vmf_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"ESN global joint PMI-proxy RL throughput  : {rl_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Reward mode                   : {rl_cfg.reward_mode}")
    print(f"ESN global joint PMI-proxy RL reward      : {rl_results['reward'].mean():.4f}")
    print(f"ESN global throughput delta       : {rl_results['rate_delta'].mean():.4f}")
    print(f"Mean proxy-SINR ratio to ZF   : {rl_results['proxy_sinr_ratio'].mean():.4f}")
    print(f"Mean actual SINR ratio to ZF  : {rl_results['actual_sinr_ratio'].mean():.4f}")
    print(f"Mean phase-invariant alignment: {rl_results['phase_alignment'].mean():.4f}")
    print(f"RL beats ZF fraction          : {rl_results['beat_zf'].mean():.4f}")
    print(f"Mean beam similarity to ZF    : {rl_results['beam_similarity_to_zf'].mean():.4f}")
    print(f"Mean grad norm                : {rl_results['grad_norm'].mean():.4f}")
    print(f"Mean PMI proxy SINR used in global shaping: {rl_results['pmi_shaping_sinr'].mean():.4f}")
    print(f"Mean PMI proxy sum-rate used in global shaping: {rl_results['pmi_shaping_sum_rate'].mean():.4f}")
    print(f"Global conditional candidates       : {rl_results['conditional_candidates'].mean():.4f}")
    print(f"Best-of-N global samples       : {rl_cfg.best_of_n}")
    print(f"Mean selected candidate index : {rl_results['best_of_n_selected'].mean():.4f}")
    print(f"Global proxy eta              : {rl_cfg.global_proxy_eta:.4f}")
    print(f"Global Gibbs sweeps           : {rl_cfg.global_gibbs_sweeps}")
    print(f"Global conditional candidates : {rl_cfg.global_conditional_candidates}")
    print(f"Global softmax temperature    : {rl_cfg.global_softmax_temperature:.4f}")


if __name__ == "__main__":
    main()
