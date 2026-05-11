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
    fixed_kappa: float = 3.0
    reward_baseline_beta: float = 0.99
    advantage_clip: float = 1.0
    grad_clip_norm: float = 1.0
    init_scale_out: float = 1e-2
    batch_size: int = 512
    reservoir_size: int = 128
    spectral_radius: float = 0.8
    input_scale: float = 0.15
    # Number of PMI time steps included in the WESN skip/readout connection.
    # A value of 1 gives the one-step skip feature [Re{z_t}, Im{z_t}, y_t].
    skip_window_length: int = 4
    # Deprecated/unused for SLNR-ratio shaping; denominator now uses noise power.
    leakage_lambda: float = 1.0
    signal_gamma: float = 0.5
    max_fb_resamples: int = 16
    leakage_norm_eps: float = 1e-12
    signal_norm_eps: float = 1e-12
    reward_mode: str = "rate_log_ratio"
    reward_reference: str = "slnr"
    reward_sinr_eps: float = 1e-12
    # Keep this at 1 for exact on-policy training. The per-UE candidate
    # sampler already uses max_fb_resamples candidate directions.
    best_of_n: int = 1
    # Candidate-softmax controls for the phase-invariant finite policy.
    # Smaller temperature makes the categorical selection stricter.
    candidate_temperature: float = 0.8
    normalize_candidate_scores: bool = True
    # Mixed candidate pool. Uniform/PMI/SLNR candidates are independent of
    # W_out. The optional WESN-centered group is generated around the current
    # WESN mean mu, giving the learned policy a chance to propose new beam
    # directions rather than only rank hand-designed PMI/SLNR candidates.
    # Fractions are converted into counts that sum to max_fb_resamples.
    uniform_candidate_frac: float = 0.10
    pmi_candidate_frac: float = 0.35
    slnr_candidate_frac: float = 0.35
    wesn_candidate_frac: float = 0.20
    candidate_proposal_kappa: float = 50.0
    # SLNR-residual mode: the learned WESN vector no longer defines an
    # independent beam center.  Instead, it defines a residual/tangent
    # correction around the PMI-SLNR direction, so SLNR is the default action
    # and WESN learns how to deform it.
    slnr_residual_scale: float = 0.15
    include_exact_anchor_candidates: bool = True
    # Auxiliary joint PMI-only full-precoder ranking loss.  This uses multiple
    # unexecuted candidate combinations through a PMI-only SINR/rate proxy,
    # while the RL term still uses only the actually executed precoder reward.
    # Set joint_pmi_aux_loss_weight=0 to disable.
    joint_pmi_aux_loss_weight: float = 0.0
    enable_oracle_pool_diagnostics: bool = False
    joint_pmi_aux_temperature: float = 0.5
    num_joint_pmi_aux_candidates: int = 32
    # Slot-level hybrid mixture policy. The DK branch executes the exact cached
    # PMI-SLNR precoder. The learned branch executes the current WESN
    # SLNR-perturbation candidate policy. Both this reuse probability and W_out
    # are trained by the policy-gradient loss.
    initial_learned_policy_probability: float = 0.25
    lr_mixture: float = 3e-2
    # Extra reward shaping for rare cases where the executed action beats the
    # selected reference baseline.  The bonus is
    #   lambda * max(rate - ref_rate, 0)^p
    # with default p=0.5, which amplifies small positive rate deltas.
    positive_rate_bonus_lambda: float = 2.0
    positive_rate_bonus_power: float = 0.5
    # Structured leakage-correction candidate policy. This replaces the old
    # random spherical-cap/vMF proposal pool with candidates of the form
    #   v_k(alpha)=normalize(v_k^SLNR + alpha u_k), alpha >= 0,
    # where u_k is the tangent direction that locally reduces PMI leakage.
    structured_leakage_alpha_grid: tuple[float, ...] = (
        0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00
    )
    # Direct-alpha policy: W_out outputs categorical logits over the
    # structured alpha candidates directly, instead of outputting a beam-center
    # mu that indirectly ranks candidates through |mu^H v(alpha)|^2.
    alpha_slnr_prior_weight: float = 0.0


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




def build_slnr_precoder_from_pmi(
    full_vk_list: list[np.ndarray],
    streams_per_user: int,
    total_tx_power: float,
    noise_power: float,
) -> np.ndarray:
    """Build a PMI-only SLNR precoder.

    For each UE k, only the dominant PMI/right-singular-vector direction q_k is
    used.  The beam direction is obtained from the rank-one SLNR problem

        max_v |q_k^H v|^2 / (sum_{j != k} |q_j^H v|^2 + sigma_eff^2),

    where sigma_eff^2 is the noise power normalized by the equal per-user
    transmit power.  This keeps the baseline comparable to the ZF-from-PMI
    baseline: both use only PMI directions, not the full channel matrix.
    """
    if streams_per_user != 1:
        raise ValueError("PMI-only SLNR baseline currently assumes one stream per UE.")

    q_vectors = []
    for vk in full_vk_list:
        q = vk[:, 0]
        q_vectors.append(q / max(np.linalg.norm(q), 1e-12))

    num_users = len(q_vectors)
    num_tx_antennas = q_vectors[0].size
    per_user_tx_power = total_tx_power / max(num_users, 1)
    sigma_eff = noise_power / max(per_user_tx_power, 1e-12)

    directions = np.zeros((num_tx_antennas, num_users), dtype=np.complex128)
    eye = np.eye(num_tx_antennas, dtype=np.complex128)
    for k, qk in enumerate(q_vectors):
        leakage = sigma_eff * eye.copy()
        for j, qj in enumerate(q_vectors):
            if j != k:
                leakage += np.outer(qj, qj.conj())

        # For rank-one desired signal q_k q_k^H, the dominant generalized
        # eigenvector is proportional to (leakage + sigma_eff I)^{-1} q_k.
        try:
            vk_slnr = np.linalg.solve(leakage, qk)
        except np.linalg.LinAlgError:
            vk_slnr = np.linalg.pinv(leakage) @ qk
        directions[:, k] = vk_slnr / max(np.linalg.norm(vk_slnr), 1e-12)

    return normalize_columns_equal_power(directions, total_tx_power)

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
      - rate_log_ratio: sum_k log((1 + SINR_RL,k + eps)/(1 + SINR_ZF,k + eps)).
      - sinr_proxy_log_ratio: same log-ratio but using the simpler SINR proxy.

    The default is rate_log_ratio so the policy is trained using a smooth
    objective that directly matches sum-rate improvement over ZF.
    """
    if reward_mode == "throughput_delta":
        return float(rate - zf_rate)
    if reward_mode == "normalized_throughput_delta":
        return float((rate - zf_rate) / (abs(zf_rate) + eps))
    if reward_mode == "actual_sinr_log_ratio":
        return float(np.mean(np.log((actual_sinr + eps) / (zf_actual_sinr + eps))))
    if reward_mode == "rate_log_ratio":
        return float(np.sum(np.log((1.0 + actual_sinr + eps) / (1.0 + zf_actual_sinr + eps))))
    if reward_mode == "sinr_proxy_log_ratio":
        return float(np.mean(np.log((proxy_sinr + eps) / (zf_proxy_sinr + eps))))
    raise ValueError(
        f"Unknown reward_mode={reward_mode!r}. Expected one of: "
        "throughput_delta, normalized_throughput_delta, "
        "actual_sinr_log_ratio, rate_log_ratio, sinr_proxy_log_ratio."
    )



def add_positive_rate_delta_bonus(
    base_reward: float,
    rate: float,
    ref_rate: float,
    bonus_lambda: float,
    bonus_power: float,
) -> float:
    """Add lambda * max(rate - ref_rate, 0)^p to the base reward.

    With p < 1, small positive improvements over the reference are amplified
    instead of suppressed.  For the exact SLNR DK branch under an SLNR
    reference, rate == ref_rate, so the bonus is exactly zero.
    """
    positive_delta = max(float(rate) - float(ref_rate), 0.0)
    if positive_delta <= 0.0 or bonus_lambda <= 0.0:
        return float(base_reward)
    p = max(float(bonus_power), 1e-12)
    return float(base_reward + float(bonus_lambda) * (positive_delta ** p))


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



def run_slnr_baseline(cfg: SimConfig, channels: np.ndarray) -> dict[str, np.ndarray]:
    """Run a PMI-only SLNR precoding baseline.

    This baseline uses the same PMI/right-singular-vector feedback as the ZF
    baseline, but designs each beam by maximizing a PMI-only SLNR criterion.
    """
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    sinr_trace = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.float64)
    precoders = np.zeros(
        (cfg.num_slots, cfg.num_tx_antennas, cfg.num_users), dtype=np.complex128
    )

    for t in range(cfg.num_slots):
        print(f"SLNR Slot {t + 1} / {cfg.num_slots}", end="\r")
        _, vk_list = pmi_features_from_channels(channels[t])
        p_slnr = build_slnr_precoder_from_pmi(
            vk_list, cfg.streams_per_user, cfg.total_tx_power, noise_power
        )
        throughput[t], sinr_trace[t] = compute_slot_sum_rate(channels[t], p_slnr, noise_power)
        precoders[t] = p_slnr

    print()
    return {
        "throughput": throughput,
        "sinr": sinr_trace,
        "precoders": precoders,
    }


def unit_norm(x: np.ndarray) -> np.ndarray:
    return x / max(np.linalg.norm(x), 1e-12)


def probability_to_logit(p: float, eps: float = 1e-6) -> float:
    """Convert a probability in (0,1) to a numerically safe logit."""
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


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


def sample_uniform_real_unit_sphere(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from the real unit sphere S^{dim-1}.

    This proposal is independent of the trainable policy parameters.  That
    matters for the candidate-based policy-gradient update below: conditioned
    on the sampled candidate set, only the categorical choice among candidates
    depends on W_out through the phase-invariant policy score.
    """
    return unit_norm(rng.standard_normal(dim))


def complex_unit_to_real(q: np.ndarray) -> np.ndarray:
    """Pack a unit-norm complex vector q as [Re{q}, Im{q}]."""
    q = q / max(np.linalg.norm(q), 1e-12)
    return np.concatenate([np.real(q), np.imag(q)]).astype(np.float64)


def apply_random_global_phase_to_real_beam(
    x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Randomly rotate a real-packed complex beam by exp(j phi)."""
    nt = x.size // 2
    v = x[:nt] + 1j * x[nt:]
    phi = rng.uniform(0.0, 2.0 * np.pi)
    v_rot = np.exp(1j * phi) * v
    return unit_norm(np.concatenate([np.real(v_rot), np.imag(v_rot)]).astype(np.float64))


def sample_phase_randomized_vmf(
    center: np.ndarray, kappa: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample a real-packed beam near center, then remove absolute phase bias."""
    x = sample_vmf(unit_norm(center), kappa, rng)
    return apply_random_global_phase_to_real_beam(x, rng)


def generalized_slnr_direction_real(
    signal_mat: np.ndarray,
    leakage_mat: np.ndarray,
    noise_floor: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return a PMI-SLNR heuristic direction independent of W_out.

    This approximates argmax_x x^T G x / (x^T L x + noise_floor) by the
    dominant generalized eigenvector of (G, L + noise_floor I).
    """
    d = signal_mat.shape[0]
    denom = leakage_mat + max(noise_floor, eps) * np.eye(d)
    try:
        mat = np.linalg.solve(denom, signal_mat)
    except np.linalg.LinAlgError:
        mat = np.linalg.pinv(denom) @ signal_mat
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    return unit_norm(np.real(eigvecs[:, int(np.argmax(eigvals))]).astype(np.float64))


def mixed_candidate_counts(
    total: int,
    uniform_frac: float,
    pmi_frac: float,
    slnr_frac: float,
    wesn_frac: float,
) -> tuple[int, int, int, int]:
    """Convert candidate fractions to integer counts summing to total."""
    total = max(1, int(total))
    fracs = np.array([uniform_frac, pmi_frac, slnr_frac, wesn_frac], dtype=np.float64)
    if np.any(fracs < 0) or float(np.sum(fracs)) <= 0.0:
        fracs = np.array([0.10, 0.35, 0.35, 0.20], dtype=np.float64)
    fracs = fracs / np.sum(fracs)
    raw = fracs * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    # Always keep at least one exploratory uniform candidate when possible.
    if counts[0] == 0 and total > 0:
        donor_candidates = [i for i in range(1, 4) if counts[i] > 1]
        if donor_candidates:
            donor = max(donor_candidates, key=lambda i: counts[i])
            counts[donor] -= 1
            counts[0] = 1
    return int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])



def parse_nonnegative_alpha_grid(s: str) -> tuple[float, ...]:
    vals: list[float] = []
    for item in s.split(","):
        item = item.strip()
        if item:
            vals.append(max(0.0, float(item)))
    if not vals:
        vals = [0.0]
    if 0.0 not in vals:
        vals = [0.0] + vals
    return tuple(sorted(set(vals)))


def positive_alpha_values_for_candidate_count(
    alpha_grid: tuple[float, ...],
    num_candidates: int,
) -> np.ndarray:
    """Return nonnegative alpha values with exactly num_candidates entries.

    If the provided alpha grid has fewer values than max_fb_resamples, fill the
    remaining entries with an evenly spaced grid over the same [0, alpha_max]
    interval. This preserves the existing [K, M, D] candidate tensor shape while
    replacing random spherical-cap proposals with deterministic structured
    leakage-step candidates.
    """
    num_candidates = max(1, int(num_candidates))
    vals = sorted({float(a) for a in alpha_grid if float(a) >= 0.0})
    if 0.0 not in vals:
        vals = [0.0] + vals
    if not vals:
        vals = [0.0]

    if len(vals) >= num_candidates:
        if num_candidates == 1:
            return np.array([0.0], dtype=np.float64)
        idx = np.linspace(0, len(vals) - 1, num_candidates)
        idx = np.unique(np.round(idx).astype(int))
        selected = [vals[i] for i in idx]
        while len(selected) < num_candidates:
            for v in vals:
                if v not in selected:
                    selected.append(v)
                    break
        return np.array(sorted(selected[:num_candidates]), dtype=np.float64)

    alpha_max = max(vals)
    if alpha_max <= 0.0:
        return np.zeros(num_candidates, dtype=np.float64)

    dense = list(np.linspace(0.0, alpha_max, num_candidates))
    merged = sorted(set(vals + [float(x) for x in dense]))
    idx = np.linspace(0, len(merged) - 1, num_candidates)
    idx = np.unique(np.round(idx).astype(int))
    selected = [merged[i] for i in idx]
    while len(selected) < num_candidates:
        for v in merged:
            if v not in selected:
                selected.append(v)
                break
    return np.array(sorted(selected[:num_candidates]), dtype=np.float64)


def leakage_reducing_tangent_direction_real(
    slnr_center: np.ndarray,
    leakage_mat: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray | None:
    """Return unit-norm tangent direction that locally reduces PMI leakage.

    In the real-packed representation, PMI leakage is L(x)=x^T L x.  The
    leakage-increasing gradient is Lx.  The tangent component that changes the
    beam direction, while ignoring pure radial changes, is

        g_perp = (I - x x^T) L x.

    The leakage-reducing tangent direction is -g_perp / ||g_perp||.
    """
    x = unit_norm(slnr_center)
    grad = leakage_mat @ x
    tangent_grad = grad - x * float(np.dot(x, grad))
    nrm = float(np.linalg.norm(tangent_grad))
    if nrm < eps:
        return None
    return -tangent_grad / nrm

def normalize_scores_np(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (scores - np.mean(scores)) / max(float(np.std(scores)), eps)


def complex_inner_abs2_from_real_np(x: np.ndarray, mu: np.ndarray) -> float:
    """Return |mu^H v|^2 for x=[Re{v}, Im{v}], mu=[Re{mu}, Im{mu}]."""
    nt = x.size // 2
    x_re = x[:nt]
    x_im = x[nt:]
    mu_re = mu[:nt]
    mu_im = mu[nt:]

    inner_re = float(np.dot(mu_re, x_re) + np.dot(mu_im, x_im))
    inner_im = float(np.dot(mu_re, x_im) - np.dot(mu_im, x_re))
    return inner_re**2 + inner_im**2




def phase_align_real_beam_to_reference_np(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotate complex beam x so that ref^H x has zero phase.

    Both x and ref are real-packed complex unit vectors [Re{v}, Im{v}].
    This is important when combining a learned correction with an SLNR anchor:
    without phase alignment, the same physical beam with a different global
    phase could destructively cancel the anchor.
    """
    nt = x.size // 2
    v = x[:nt] + 1j * x[nt:]
    r = ref[:nt] + 1j * ref[nt:]
    inner = np.vdot(r, v)
    if np.abs(inner) < 1e-12:
        return unit_norm(x.copy())
    v_aligned = np.exp(-1j * np.angle(inner)) * v
    return unit_norm(np.concatenate([np.real(v_aligned), np.imag(v_aligned)]).astype(np.float64))


def build_slnr_residual_center_np(
    mu: np.ndarray, slnr_center: np.ndarray, residual_scale: float
) -> np.ndarray:
    """Build normalize(v_SLNR + alpha * Delta(mu)) in real-packed form.

    The learned WESN output mu is interpreted as a residual direction around
    the SLNR anchor, not as an independent beam.  We first align mu's global
    phase to the SLNR direction, remove the component parallel to SLNR, and
    then add the remaining tangent-like correction to the SLNR anchor.
    """
    slnr_center = unit_norm(slnr_center)
    mu_aligned = phase_align_real_beam_to_reference_np(unit_norm(mu), slnr_center)
    parallel = float(np.dot(slnr_center, mu_aligned)) * slnr_center
    residual = mu_aligned - parallel
    if np.linalg.norm(residual) < 1e-12:
        return slnr_center.copy()
    return unit_norm(slnr_center + float(residual_scale) * residual)


def build_slnr_residual_center_torch(
    mu: torch.Tensor, slnr_center: torch.Tensor, residual_scale: float
) -> torch.Tensor:
    """Torch version of build_slnr_residual_center_np.

    Shapes: mu and slnr_center are [..., D] real-packed complex vectors.
    """
    d = mu.shape[-1]
    nt = d // 2
    mu = mu / torch.clamp(torch.linalg.norm(mu, dim=-1, keepdim=True), min=1e-12)
    slnr_center = slnr_center / torch.clamp(torch.linalg.norm(slnr_center, dim=-1, keepdim=True), min=1e-12)

    mu_re, mu_im = mu[..., :nt], mu[..., nt:]
    ref_re, ref_im = slnr_center[..., :nt], slnr_center[..., nt:]

    inner_re = torch.sum(ref_re * mu_re + ref_im * mu_im, dim=-1, keepdim=True)
    inner_im = torch.sum(ref_re * mu_im - ref_im * mu_re, dim=-1, keepdim=True)
    mag = torch.clamp(torch.sqrt(inner_re**2 + inner_im**2), min=1e-12)

    # Multiply mu by exp(-j angle(ref^H mu)).
    cos_phi = inner_re / mag
    sin_phi = -inner_im / mag
    mu_aligned_re = cos_phi * mu_re - sin_phi * mu_im
    mu_aligned_im = sin_phi * mu_re + cos_phi * mu_im
    mu_aligned = torch.cat([mu_aligned_re, mu_aligned_im], dim=-1)

    parallel_coeff = torch.sum(slnr_center * mu_aligned, dim=-1, keepdim=True)
    residual = mu_aligned - parallel_coeff * slnr_center
    center = slnr_center + float(residual_scale) * residual
    return center / torch.clamp(torch.linalg.norm(center, dim=-1, keepdim=True), min=1e-12)


def complex_inner_abs2_from_real_torch(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Return |mu^H v|^2 for real-packed complex vectors.

    x can have an additional candidate dimension, e.g. [B, K, M, D], while
    mu can be [B, K, 1, D] or [B, K, D]. Broadcasting is supported.
    """
    d = x.shape[-1]
    nt = d // 2

    x_re = x[..., :nt]
    x_im = x[..., nt:]
    mu_re = mu[..., :nt]
    mu_im = mu[..., nt:]

    inner_re = torch.sum(mu_re * x_re + mu_im * x_im, dim=-1)
    inner_im = torch.sum(mu_re * x_im - mu_im * x_re, dim=-1)
    return inner_re**2 + inner_im**2


def alpha_candidate_log_probs_torch(
    alpha_logits: torch.Tensor,
    slnr_scores: torch.Tensor,
    candidate_temperature: float,
    normalize_candidate_scores: bool,
    alpha_slnr_prior_weight: float = 0.0,
) -> torch.Tensor:
    """Log-probabilities for the direct-alpha structured policy.

    alpha_logits has shape [B, K, M] and is produced directly by W_out.
    Candidate i corresponds to alpha_i in the deterministic structured
    leakage-correction grid.  This replaces the earlier indirect beam-center
    policy score kappa |mu^H v(alpha_i)|^2.

    If alpha_slnr_prior_weight > 0, a normalized PMI-SLNR candidate score is
    added as a fixed prior.  The default is zero so WESN learns the alpha
    categorical distribution directly.
    """
    logits = alpha_logits
    if alpha_slnr_prior_weight != 0.0:
        if normalize_candidate_scores:
            prior = (
                slnr_scores - torch.mean(slnr_scores, dim=-1, keepdim=True)
            ) / torch.clamp(torch.std(slnr_scores, dim=-1, keepdim=True, unbiased=False), min=1e-12)
        else:
            prior = slnr_scores
        logits = logits + float(alpha_slnr_prior_weight) * prior
    logits = logits / max(float(candidate_temperature), 1e-12)
    return torch.log_softmax(logits, dim=-1)


def phase_invariant_candidate_log_probs_torch(
    candidate_pool: torch.Tensor,
    mu: torch.Tensor,
    fixed_kappa: float,
    slnr_scores: torch.Tensor,
    candidate_temperature: float,
    normalize_candidate_scores: bool,
    slnr_centers: torch.Tensor | None = None,
    slnr_residual_scale: float = 0.0,
) -> torch.Tensor:
    """Log-probabilities over all candidates under the finite policy.

    candidate_pool has shape [B, K, M, D], mu has shape [B, K, D], and
    slnr_scores has shape [B, K, M].  The returned tensor has shape [B, K, M].

    If slnr_centers is provided, the WESN mean is interpreted as a residual
    correction around the SLNR anchor.  The phase score then uses

        c = normalize(v_SLNR + alpha * Delta(mu))

    instead of using mu directly.
    """
    if slnr_centers is not None:
        policy_center = build_slnr_residual_center_torch(
            mu=mu,
            slnr_center=slnr_centers,
            residual_scale=slnr_residual_scale,
        )
    else:
        policy_center = mu
    mu_expanded = policy_center.unsqueeze(-2)
    phase_scores = fixed_kappa * complex_inner_abs2_from_real_torch(candidate_pool, mu_expanded)
    if normalize_candidate_scores:
        phase_scores = (
            phase_scores - torch.mean(phase_scores, dim=-1, keepdim=True)
        ) / torch.clamp(torch.std(phase_scores, dim=-1, keepdim=True, unbiased=False), min=1e-12)
        slnr_scores = (
            slnr_scores - torch.mean(slnr_scores, dim=-1, keepdim=True)
        ) / torch.clamp(torch.std(slnr_scores, dim=-1, keepdim=True, unbiased=False), min=1e-12)
    logits = (phase_scores + slnr_scores) / max(float(candidate_temperature), 1e-12)
    return torch.log_softmax(logits, dim=-1)


def phase_invariant_candidate_log_prob_torch(
    candidate_pool: torch.Tensor,
    selected_indices: torch.Tensor,
    mu: torch.Tensor,
    fixed_kappa: float,
    slnr_scores: torch.Tensor,
    candidate_temperature: float,
    normalize_candidate_scores: bool,
    slnr_centers: torch.Tensor | None = None,
    slnr_residual_scale: float = 0.0,
) -> torch.Tensor:
    """Log-probability of the selected candidate under the finite policy.

    candidate_pool has shape [B, K, M, D], selected_indices has shape [B, K],
    mu has shape [B, K, D], and slnr_scores has shape [B, K, M].

    The candidate pool may include a WESN-centered group generated from the
    current mean.  In this training loss, the realized pool is treated as fixed
    (stop-gradient proposal approximation). Conditioned on this pool, the
    categorical selection probability depends on W_out:

        pi(i | s, C) = softmax_i((phase_i + slnr_i) / tau).

    This keeps the REINFORCE update correct for the trainable candidate selector.
    """
    log_probs = phase_invariant_candidate_log_probs_torch(
        candidate_pool=candidate_pool,
        mu=mu,
        fixed_kappa=fixed_kappa,
        slnr_scores=slnr_scores,
        candidate_temperature=candidate_temperature,
        normalize_candidate_scores=normalize_candidate_scores,
        slnr_centers=slnr_centers,
        slnr_residual_scale=slnr_residual_scale,
    )
    return torch.gather(log_probs, dim=-1, index=selected_indices.unsqueeze(-1)).squeeze(-1)


def joint_pmi_proxy_aux_loss_torch(
    log_probs_all: torch.Tensor,
    candidate_pool: torch.Tensor,
    q_real: torch.Tensor,
    noise_floor: float,
    temperature: float,
    num_joint_candidates: int,
) -> torch.Tensor:
    """Joint PMI-only full-precoder auxiliary ranking loss.

    This loss samples candidate-index combinations across users, builds a
    PMI-only proxy sum-rate for each full candidate precoder, and uses those
    proxy rates as soft targets over complete multi-user candidate combinations.

    Shapes:
      log_probs_all:  [B, K, M]
      candidate_pool: [B, K, M, D] with real-packed unit-norm complex beams
      q_real:         [B, K, D] with real-packed PMI/right singular vectors

    The true environment reward is *not* used for unexecuted candidates here.
    The reward used for the auxiliary targets is the PMI-only proxy

        SINR_hat_k = |q_k^H v_k|^2 / (sum_{j != k}|q_k^H v_j|^2 + noise_floor)

    evaluated for sampled full-precoder candidate combinations.
    """
    batch_size, num_users, num_candidates = log_probs_all.shape
    j_count = max(1, int(num_joint_candidates))
    device = log_probs_all.device
    dim = candidate_pool.shape[-1]

    joint_indices = torch.randint(
        low=0,
        high=num_candidates,
        size=(batch_size, j_count, num_users),
        device=device,
    )

    # Always include the current greedy per-UE combination as one joint candidate.
    greedy_indices = torch.argmax(log_probs_all.detach(), dim=-1)  # [B, K]
    joint_indices[:, 0, :] = greedy_indices

    # Gather beams for all sampled full-precoder combinations: [B, J, K, D].
    candidate_pool_expanded = candidate_pool.unsqueeze(1).expand(
        batch_size, j_count, num_users, num_candidates, dim
    )
    gather_idx = joint_indices.unsqueeze(-1).unsqueeze(-1).expand(
        batch_size, j_count, num_users, 1, dim
    )
    selected_beams = torch.gather(
        candidate_pool_expanded, dim=3, index=gather_idx
    ).squeeze(3)

    # gains[b, rx, m, tx] = |q_{b,rx}^H v_{b,m,tx}|^2.
    q_expanded = q_real.unsqueeze(2).unsqueeze(3)  # [B, K_rx, 1, 1, D]
    v_expanded = selected_beams.unsqueeze(1)       # [B, 1, J, K_tx, D]
    gains = complex_inner_abs2_from_real_torch(v_expanded, q_expanded)

    # Diagonal terms are desired gains for each RX user under each joint candidate.
    desired = torch.diagonal(gains, dim1=1, dim2=3).permute(0, 1, 2)  # [B, J, K]
    total = torch.sum(gains, dim=-1).permute(0, 2, 1)                # [B, J, K]
    interference = torch.clamp(total - desired, min=0.0)
    proxy_sinr = desired / (interference + max(float(noise_floor), 1e-12))
    proxy_reward = torch.sum(torch.log1p(proxy_sinr), dim=-1)       # [B, J]

    centered_reward = proxy_reward - torch.mean(proxy_reward, dim=-1, keepdim=True)
    weights = torch.softmax(centered_reward / max(float(temperature), 1e-12), dim=-1).detach()

    # log pi(P^{(m)}|s,C) = sum_k log pi_k(i_k^{(m)}|s,C_k).
    log_probs_expanded = log_probs_all.unsqueeze(1).expand(
        batch_size, j_count, num_users, num_candidates
    )
    combo_log_probs = torch.gather(
        log_probs_expanded, dim=-1, index=joint_indices.unsqueeze(-1)
    ).squeeze(-1).sum(dim=-1)  # [B, J]

    return -torch.mean(torch.sum(weights * combo_log_probs, dim=-1)) / max(num_users, 1)


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
    alpha_logits: np.ndarray,
    kappa: float,
    signal_mat: np.ndarray,
    leakage_mat: np.ndarray,
    signal_gamma: float,
    slnr_noise_power: float,
    rng: np.random.Generator,
    max_resamples: int,
    signal_norm_eps: float,
    leakage_norm_eps: float,
    pmi_center: np.ndarray,
    candidate_temperature: float,
    normalize_candidate_scores: bool,
    uniform_candidate_frac: float,
    pmi_candidate_frac: float,
    slnr_candidate_frac: float,
    wesn_candidate_frac: float,
    candidate_proposal_kappa: float,
    slnr_residual_scale: float,
    include_exact_anchor_candidates: bool,
    structured_leakage_alpha_grid: tuple[float, ...],
    slnr_center_override: np.ndarray | None = None,
) -> tuple[
    np.ndarray, float, float, int, bool, np.ndarray, np.ndarray, np.ndarray, int, dict[str, float], np.ndarray
]:
    """Candidate-based structured leakage-correction PMI-SLNR policy sampler.

    The old mixed spherical-cap/vMF candidate pool is replaced by a deterministic
    one-direction structured pool.  Each candidate starts from the exact SLNR
    beam and takes a nonnegative step along the PMI leakage-reducing tangent
    direction.

    The WESN now learns alpha directly.  For each UE, W_out produces a vector
    of categorical logits over the structured alpha candidates.  Candidate i is

        v_i = normalize(v_SLNR + alpha_i u_leak).

    The categorical selector is therefore

        pi(i | s, C) ∝ exp(logit_alpha_i / tau),

    optionally plus a fixed PMI-SLNR prior if alpha_slnr_prior_weight is set
    outside this sampler.  This removes the earlier indirect beam-center score
    kappa |mu^H v_i|^2.
    """
    num_candidates = max(1, int(max_resamples))
    alpha_logits = np.asarray(alpha_logits, dtype=np.float64).reshape(-1)
    if alpha_logits.size != num_candidates:
        raise ValueError(f"alpha_logits has size {alpha_logits.size}, expected {num_candidates}")
    dim = pmi_center.size
    pmi_center = unit_norm(pmi_center)
    noise_floor = max(float(slnr_noise_power), leakage_norm_eps)

    if slnr_center_override is None:
        slnr_center = generalized_slnr_direction_real(signal_mat, leakage_mat, noise_floor)
    else:
        slnr_center = unit_norm(slnr_center_override)

    # Structured one-direction positive leakage-correction pool. Candidate 0 is
    # alpha=0, i.e., the exact SLNR anchor.  All other candidates are positive
    # steps along the leakage-reducing tangent direction.
    alphas = positive_alpha_values_for_candidate_count(
        structured_leakage_alpha_grid,
        num_candidates,
    )
    u_leak = leakage_reducing_tangent_direction_real(
        slnr_center=slnr_center,
        leakage_mat=leakage_mat,
        eps=leakage_norm_eps,
    )

    candidate_pool = np.zeros((num_candidates, dim), dtype=np.float64)
    source_pool = np.full(num_candidates, 3, dtype=np.int64)  # 2=SLNR alpha=0, 3=structured leakage step
    if u_leak is None:
        for i in range(num_candidates):
            candidate_pool[i] = slnr_center
            source_pool[i] = 2
    else:
        for i, alpha in enumerate(alphas):
            candidate_pool[i] = unit_norm(slnr_center + float(alpha) * u_leak)
            source_pool[i] = 2 if abs(float(alpha)) < 1e-14 else 3

    # Backward-compatible diagnostics.  The old mixed-proposal counts no longer
    # apply; all nonzero-alpha candidates are structured leakage-step candidates.
    n_uniform = 0
    n_pmi = 0
    n_slnr = int(np.sum(np.abs(alphas) < 1e-14))
    n_wesn = int(num_candidates - n_slnr)
    signal_pool = np.zeros(num_candidates, dtype=np.float64)
    leakage_pool = np.zeros(num_candidates, dtype=np.float64)
    phase_score_pool = np.zeros(num_candidates, dtype=np.float64)
    slnr_score_pool = np.zeros(num_candidates, dtype=np.float64)

    for i in range(num_candidates):
        x = candidate_pool[i]
        g = raw_quadratic_score(x, signal_mat)
        ell = raw_quadratic_score(x, leakage_mat)
        signal_pool[i] = g
        leakage_pool[i] = ell
        # For backward-compatible diagnostics, store the direct WESN alpha logit
        # in phase_score_pool.  It is no longer a phase/beam-center score.
        phase_score_pool[i] = alpha_logits[i]
        slnr_score_pool[i] = signal_gamma * g / max(ell + noise_floor, leakage_norm_eps)

    # Direct-alpha policy.  By default this uses only WESN alpha logits.
    # A PMI-SLNR prior can be added in the torch training path through
    # alpha_slnr_prior_weight; the numpy sampler mirrors the default direct mode.
    policy_score_pool = alpha_logits.copy()
    logits = policy_score_pool / max(float(candidate_temperature), leakage_norm_eps)
    stable_logits = logits - np.max(logits)
    probs = np.exp(stable_logits)
    probs = probs / max(float(np.sum(probs)), leakage_norm_eps)
    selected_idx = int(rng.choice(num_candidates, p=probs))

    x_selected = candidate_pool[selected_idx]
    empirical_mean = np.sum(probs[:, None] * candidate_pool, axis=0)

    entropy = -float(np.sum(probs * np.log(np.maximum(probs, leakage_norm_eps))))
    eff_num = 1.0 / max(float(np.sum(probs**2)), leakage_norm_eps)
    diag = {
        "entropy": entropy,
        "entropy_norm": entropy / max(np.log(num_candidates), leakage_norm_eps),
        "effective_candidates": eff_num,
        "selected_policy_score": float(policy_score_pool[selected_idx]),
        "best_policy_score": float(np.max(policy_score_pool)),
        "mean_policy_score": float(np.mean(policy_score_pool)),
        "selected_minus_mean_policy_score": float(policy_score_pool[selected_idx] - np.mean(policy_score_pool)),
        "phase_score_mean": float(np.mean(phase_score_pool)),
        "phase_score_std": float(np.std(phase_score_pool)),
        "slnr_score_mean": float(np.mean(slnr_score_pool)),
        "slnr_score_std": float(np.std(slnr_score_pool)),
        "policy_score_std": float(np.std(policy_score_pool)),
        "selected_source": float(source_pool[selected_idx]),
        "uniform_count": float(n_uniform),
        "pmi_count": float(n_pmi),
        "slnr_count": float(n_slnr),
        "wesn_count": float(n_wesn),
    }

    # attempts is retained as a diagnostic field. It stores the 1-based selected
    # candidate index; accepted is always True because this is a selector, not an
    # accept/reject sampler.
    return (
        x_selected,
        float(signal_pool[selected_idx]),
        float(leakage_pool[selected_idx]),
        selected_idx + 1,
        True,
        empirical_mean,
        candidate_pool,
        slnr_score_pool,
        selected_idx,
        diag,
        slnr_center,
    )

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


def build_windowed_skip_features(pmi_features: np.ndarray, window_length: int) -> np.ndarray:
    """Build padded input windows for WESN skip/readout connections.

    pmi_features has shape [T, D_y], where each row y_t is already real-valued
    because it contains [Re{PMI}, Im{PMI}]. The returned array has shape
    [T, K_skip * D_y]. At early slots, the first available input is repeated on
    the left, matching the padding convention used by configured_wesn_pred.py.

    For K_skip=3, the skip feature at time t is [y_{t-2}, y_{t-1}, y_t],
    with left padding for t < 2.
    """
    num_slots, feat_dim = pmi_features.shape
    k_skip = max(1, int(window_length))

    if k_skip == 1:
        return pmi_features.astype(np.float64, copy=True)

    out = np.zeros((num_slots, k_skip * feat_dim), dtype=np.float64)
    for t in range(num_slots):
        start = max(0, t - k_skip + 1)
        win = pmi_features[start : t + 1]
        if win.shape[0] < k_skip:
            pad = np.repeat(win[0:1], k_skip - win.shape[0], axis=0)
            win = np.concatenate([pad, win], axis=0)
        out[t, :] = win.reshape(-1)
    return out


def compute_wesn_states(pmi_features: np.ndarray, rl_cfg: RLConfig, rng: np.random.Generator) -> np.ndarray:
    """Compute fixed WESN readout features from all-UE PMI features.

    The reservoir update is the same as the vanilla ESN,

        z_t = tanh(W_in y_t + W_res z_{t-1}),

    but the readout feature uses a windowed skip connection,

        s_t = [Re{z_t}, Im{z_t}, y_{t-K+1}, ..., y_t],

    where K = rl_cfg.skip_window_length. In this script, y_t is already
    real-valued because pmi_features_from_channels() concatenates Re{V_k} and
    Im{V_k}, so the windowed skip block appends these real vectors directly.
    Only W_out is trained by the RL algorithm.
    """
    num_slots, feat_dim = pmi_features.shape
    nz = rl_cfg.reservoir_size
    k_skip = max(1, int(rl_cfg.skip_window_length))
    w_in, w_res = make_fixed_esn(
        feat_dim=feat_dim,
        reservoir_size=nz,
        spectral_radius=rl_cfg.spectral_radius,
        input_scale=rl_cfg.input_scale,
        rng=rng,
    )

    skip_features = build_windowed_skip_features(pmi_features, k_skip)

    z = np.zeros(nz, dtype=np.complex128)
    states = np.zeros((num_slots, 2 * nz + k_skip * feat_dim), dtype=np.float64)
    for t in range(num_slots):
        y = pmi_features[t]
        z = split_tanh_np(w_in @ y + w_res @ z)
        states[t, :] = np.concatenate([np.real(z), np.imag(z), skip_features[t]])
    return states


def vmf_log_prob_fixed_kappa_torch(
    x: torch.Tensor, mu: torch.Tensor, fixed_kappa: float
) -> torch.Tensor:
    """vMF log probability up to a constant when kappa is fixed."""
    return fixed_kappa * torch.sum(mu * x, dim=-1)


def run_wesn_policy_rl(
    cfg: SimConfig,
    rl_cfg: RLConfig,
    channels: np.ndarray,
    zf_baseline: dict[str, np.ndarray],
    slnr_baseline: dict[str, np.ndarray],
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
    # is augmented with a windowed skip connection from recent PMI inputs.
    wesn_states = compute_wesn_states(zf_baseline["pmi_features"], rl_cfg, rng)
    state_dim = wesn_states.shape[1]

    # W_out maps shared WESN readout features directly to per-UE alpha logits,
    # shape [K, M, 2*N_z + skip_window_length*input_dim], where M is the
    # number of structured leakage-step alpha candidates.  This replaces the
    # earlier high-dimensional beam-center output mu in R^{2N_t}.
    num_alpha_candidates = max(1, int(rl_cfg.max_fb_resamples))
    w_out_init = rl_cfg.init_scale_out * rng.standard_normal((k, num_alpha_candidates, state_dim))
    w_out = torch.nn.Parameter(torch.tensor(w_out_init, dtype=torch.float64))

    # Trainable slot-level mixture/reuse probability.
    #   branch = DK_SLNR with probability 1 - sigmoid(mixture_logit)
    #   branch = learned SLNR-perturbation with probability sigmoid(mixture_logit)
    mixture_logit = torch.nn.Parameter(
        torch.tensor(
            probability_to_logit(rl_cfg.initial_learned_policy_probability),
            dtype=torch.float64,
        )
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [w_out], "lr": rl_cfg.lr_out},
            {"params": [mixture_logit], "lr": rl_cfg.lr_mixture},
        ]
    )

    if rl_cfg.best_of_n != 1:
        raise ValueError(
            "Exact on-policy training with the candidate-based phase-invariant "
            "sampler requires --best-of-n 1. Use --max-fb-resamples to control "
            "the number of candidate beam directions per UE."
        )

    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    reward_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    advantage_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    baseline_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    kappa_trace = np.full((cfg.num_slots, k), rl_cfg.fixed_kappa, dtype=np.float64)
    beat_zf_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    beat_slnr_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    rate_delta_slnr_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    actual_sinr_ratio_slnr_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    proxy_sinr_ratio_slnr_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    oracle_slnr_anchor_rate_trace = np.full(cfg.num_slots, np.nan, dtype=np.float64)
    oracle_best_pool_rate_trace = np.full(cfg.num_slots, np.nan, dtype=np.float64)
    beam_similarity_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    grad_norm_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    loss_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    rl_loss_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    joint_pmi_aux_loss_trace = np.zeros(cfg.num_slots, dtype=np.float64)
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
    candidate_entropy_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_entropy_norm_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_effective_count_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_selected_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_best_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_mean_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_selected_minus_mean_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_phase_score_mean_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_phase_score_std_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_slnr_score_mean_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_slnr_score_std_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_policy_score_std_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_selected_source_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    learned_policy_probability_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    dk_policy_probability_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    selected_learned_branch_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    selected_dk_branch_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    mixture_logit_trace = np.zeros(cfg.num_slots, dtype=np.float64)

    reward_baseline: float | None = None
    batch_s: list[np.ndarray] = []
    batch_candidate_pools: list[np.ndarray] = []
    batch_slnr_score_pools: list[np.ndarray] = []
    batch_selected_indices: list[np.ndarray] = []
    batch_slnr_centers: list[np.ndarray] = []
    batch_q_real: list[np.ndarray] = []
    batch_branch_is_learned: list[float] = []
    batch_adv: list[float] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        nonlocal batch_s, batch_candidate_pools, batch_slnr_score_pools, batch_selected_indices, batch_slnr_centers, batch_q_real, batch_branch_is_learned, batch_adv, batch_indices
        if not batch_s:
            return

        s_batch = torch.tensor(np.stack(batch_s, axis=0), dtype=torch.float64)
        candidate_pool_batch = torch.tensor(np.stack(batch_candidate_pools, axis=0), dtype=torch.float64)
        slnr_score_pool_batch = torch.tensor(np.stack(batch_slnr_score_pools, axis=0), dtype=torch.float64)
        selected_index_batch = torch.tensor(np.stack(batch_selected_indices, axis=0), dtype=torch.long)
        slnr_center_batch = torch.tensor(np.stack(batch_slnr_centers, axis=0), dtype=torch.float64)
        q_real_batch = torch.tensor(np.stack(batch_q_real, axis=0), dtype=torch.float64)
        branch_is_learned_batch = torch.tensor(np.array(batch_branch_is_learned), dtype=torch.float64)
        adv_batch = torch.tensor(np.array(batch_adv), dtype=torch.float64)

        # Recompute log probabilities under the current W_out for REINFORCE.
        # Direct-alpha logits shape: [B, K, M], where M indexes the structured
        # leakage-step alpha candidates.
        alpha_logits = torch.einsum("kmn,bn->bkm", w_out, s_batch)

        # Exact conditional log-probability for the direct-alpha structured
        # policy.  The realized candidate pools are fixed; W_out directly scores
        # alpha candidates instead of producing a beam center mu.
        log_probs_all = alpha_candidate_log_probs_torch(
            alpha_logits=alpha_logits,
            slnr_scores=slnr_score_pool_batch,
            candidate_temperature=rl_cfg.candidate_temperature,
            normalize_candidate_scores=rl_cfg.normalize_candidate_scores,
            alpha_slnr_prior_weight=rl_cfg.alpha_slnr_prior_weight,
        )
        per_user_log_prob = torch.gather(
            log_probs_all, dim=-1, index=selected_index_batch.unsqueeze(-1)
        ).squeeze(-1)
        learned_branch_log_prob = torch.sum(per_user_log_prob, dim=1)

        # Exact two-branch hybrid policy log-probability. The DK branch is the
        # deterministic exact cached SLNR precoder, so its conditional action
        # log-probability is zero; only the branch reuse probability contributes.
        p_learned = torch.sigmoid(mixture_logit)
        log_p_learned = torch.log(torch.clamp(p_learned, min=1e-12))
        log_p_dk = torch.log(torch.clamp(1.0 - p_learned, min=1e-12))
        branch_log_prob = branch_is_learned_batch * log_p_learned + (1.0 - branch_is_learned_batch) * log_p_dk
        joint_log_prob = branch_log_prob + branch_is_learned_batch * learned_branch_log_prob
        rl_loss = -torch.mean(adv_batch * joint_log_prob)

        # Joint PMI-only full-precoder auxiliary loss.  This ranks sampled
        # complete candidate precoders using a PMI-only proxy sum-rate.  No
        # additional environment interaction is used; unexecuted candidates get
        # proxy targets only, while rl_loss remains driven by the actually
        # executed precoder reward.
        if rl_cfg.joint_pmi_aux_loss_weight > 0.0:
            joint_pmi_aux_loss = joint_pmi_proxy_aux_loss_torch(
                log_probs_all=log_probs_all,
                candidate_pool=candidate_pool_batch,
                q_real=q_real_batch,
                noise_floor=slnr_noise_power,
                temperature=rl_cfg.joint_pmi_aux_temperature,
                num_joint_candidates=rl_cfg.num_joint_pmi_aux_candidates,
            )
            loss = rl_loss + rl_cfg.joint_pmi_aux_loss_weight * joint_pmi_aux_loss
        else:
            joint_pmi_aux_loss = torch.zeros((), dtype=torch.float64)
            loss = rl_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([w_out, mixture_logit], rl_cfg.grad_clip_norm)
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        rl_loss_value = float(rl_loss.detach().cpu().item())
        joint_pmi_aux_loss_value = float(joint_pmi_aux_loss.detach().cpu().item())
        grad_norm_value = float(grad_norm.detach().cpu().item())
        for idx in batch_indices:
            loss_trace[idx] = loss_value
            rl_loss_trace[idx] = rl_loss_value
            joint_pmi_aux_loss_trace[idx] = joint_pmi_aux_loss_value
            grad_norm_trace[idx] = grad_norm_value

        batch_s = []
        batch_candidate_pools = []
        batch_slnr_score_pools = []
        batch_selected_indices = []
        batch_slnr_centers = []
        batch_q_real = []
        batch_branch_is_learned = []
        batch_adv = []
        batch_indices = []

    for t in range(cfg.num_slots):
        print(f"WESN-RL Slot {t + 1} / {cfg.num_slots}", end="\r")
        s = wesn_states[t]

        # All users' direct alpha logits are produced from the same shared WESN
        # readout feature.  Shape: [K, M].
        s_t = torch.tensor(s, dtype=torch.float64)
        with torch.no_grad():
            alpha_logits_t = torch.einsum("kmn,n->km", w_out, s_t)
            alpha_logits_np = alpha_logits_t.detach().cpu().numpy()

        # Emulate the limited-feedback information available to the policy.
        # The sampler uses only dominant right singular vectors / PMI directions,
        # not full channel Gram matrices H_k^H H_k.
        q_vectors = pmi_dominant_vectors_from_channels(channels[t])
        q_real_sample = np.stack([complex_unit_to_real(q_vectors[ku]) for ku in range(k)], axis=0)
        signal_mats, leakage_mats = build_pmi_signal_and_leakage_matrices_real(q_vectors)

        zf_rate = float(zf_baseline["throughput"][t])
        zf_actual_sinr = zf_baseline["sinr"][t]
        zf_proxy_sinr = compute_pmi_sinr_proxy_from_precoder(
            q_vectors, zf_baseline["precoders"][t], noise_power, eps=rl_cfg.reward_sinr_eps
        )

        slnr_precoder_t = slnr_baseline["precoders"][t]
        slnr_rate = float(slnr_baseline["throughput"][t])
        slnr_actual_sinr = slnr_baseline["sinr"][t]
        slnr_proxy_sinr = compute_pmi_sinr_proxy_from_precoder(
            q_vectors, slnr_precoder_t, noise_power, eps=rl_cfg.reward_sinr_eps
        )

        if rl_cfg.reward_reference == "slnr":
            ref_rate = slnr_rate
            ref_actual_sinr = slnr_actual_sinr
            ref_proxy_sinr = slnr_proxy_sinr
        elif rl_cfg.reward_reference == "zf":
            ref_rate = zf_rate
            ref_actual_sinr = zf_actual_sinr
            ref_proxy_sinr = zf_proxy_sinr
        else:
            raise ValueError(f"Unknown reward_reference={rl_cfg.reward_reference!r}")

        # Slot-level branch-first hybrid mixture policy.
        # DK branch: exact cached PMI-SLNR precoder.
        # Learned branch: current WESN SLNR-perturbation candidate policy.
        with torch.no_grad():
            p_learned_np = float(torch.sigmoid(mixture_logit).detach().cpu().item())
        slot_from_learned = bool(rng.uniform(0.0, 1.0) < p_learned_np)
        learned_policy_probability_trace[t] = p_learned_np
        dk_policy_probability_trace[t] = 1.0 - p_learned_np
        selected_learned_branch_trace[t] = 1.0 if slot_from_learned else 0.0
        selected_dk_branch_trace[t] = 1.0 - selected_learned_branch_trace[t]
        mixture_logit_trace[t] = float(mixture_logit.detach().cpu().item())

        if slot_from_learned:
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
            best_candidate_pool = None
            best_slnr_score_pool = None
            best_selected_indices = None
            best_slnr_centers = None
            best_diag = None
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
                cand_candidate_pool = np.zeros((k, max(1, rl_cfg.max_fb_resamples), d), dtype=np.float64)
                cand_slnr_score_pool = np.zeros((k, max(1, rl_cfg.max_fb_resamples)), dtype=np.float64)
                cand_selected_indices = np.zeros(k, dtype=np.int64)
                cand_slnr_centers = np.zeros((k, d), dtype=np.float64)
                cand_diag = {
                    "entropy": np.zeros(k, dtype=np.float64),
                    "entropy_norm": np.zeros(k, dtype=np.float64),
                    "effective_candidates": np.zeros(k, dtype=np.float64),
                    "selected_policy_score": np.zeros(k, dtype=np.float64),
                    "best_policy_score": np.zeros(k, dtype=np.float64),
                    "mean_policy_score": np.zeros(k, dtype=np.float64),
                    "selected_minus_mean_policy_score": np.zeros(k, dtype=np.float64),
                    "phase_score_mean": np.zeros(k, dtype=np.float64),
                    "phase_score_std": np.zeros(k, dtype=np.float64),
                    "slnr_score_mean": np.zeros(k, dtype=np.float64),
                    "slnr_score_std": np.zeros(k, dtype=np.float64),
                    "policy_score_std": np.zeros(k, dtype=np.float64),
                    "selected_source": np.zeros(k, dtype=np.float64),
                }

                for ku in range(k):
                    (
                        xk,
                        sig,
                        ell,
                        attempts,
                        accepted,
                        emp_mean,
                        candidate_pool,
                        slnr_score_pool,
                        selected_idx,
                        diag,
                        slnr_residual_center,
                    ) = sample_fisher_bingham_signal_leakage(
                        alpha_logits=alpha_logits_np[ku],
                        kappa=rl_cfg.fixed_kappa,
                        signal_mat=signal_mats[ku],
                        leakage_mat=leakage_mats[ku],
                        signal_gamma=rl_cfg.signal_gamma,
                        slnr_noise_power=slnr_noise_power,
                        rng=rng,
                        max_resamples=rl_cfg.max_fb_resamples,
                        signal_norm_eps=rl_cfg.signal_norm_eps,
                        leakage_norm_eps=rl_cfg.leakage_norm_eps,
                        pmi_center=complex_unit_to_real(q_vectors[ku]),
                        candidate_temperature=rl_cfg.candidate_temperature,
                        normalize_candidate_scores=rl_cfg.normalize_candidate_scores,
                        uniform_candidate_frac=rl_cfg.uniform_candidate_frac,
                        pmi_candidate_frac=rl_cfg.pmi_candidate_frac,
                        slnr_candidate_frac=rl_cfg.slnr_candidate_frac,
                        wesn_candidate_frac=rl_cfg.wesn_candidate_frac,
                        candidate_proposal_kappa=rl_cfg.candidate_proposal_kappa,
                        slnr_residual_scale=rl_cfg.slnr_residual_scale,
                        include_exact_anchor_candidates=rl_cfg.include_exact_anchor_candidates,
                        structured_leakage_alpha_grid=rl_cfg.structured_leakage_alpha_grid,
                        slnr_center_override=complex_unit_to_real(slnr_precoder_t[:, ku]),
                    )
                    cand_x_sample[ku] = xk
                    cand_signal[ku] = sig
                    cand_leakage[ku] = ell
                    cand_attempts[ku] = attempts
                    cand_accept[ku] = 1.0 if accepted else 0.0
                    cand_emp_mean[ku] = emp_mean
                    cand_candidate_pool[ku] = candidate_pool
                    cand_slnr_score_pool[ku] = slnr_score_pool
                    cand_selected_indices[ku] = selected_idx
                    cand_slnr_centers[ku] = complex_unit_to_real(slnr_precoder_t[:, ku])
                    for diag_key in cand_diag:
                        cand_diag[diag_key][ku] = diag[diag_key]
                    cand_beams[:, ku] = real_to_complex_beam(cand_x_sample[ku], cfg.total_tx_power, k)

                cand_rate, cand_actual_sinr = compute_slot_sum_rate(
                    channels[t], cand_beams, noise_power
                )
                # The default RL reward uses the true post-detection SINR through
                # reward_mode="rate_log_ratio".  The PMI-only proxy is still
                # computed and logged for diagnostics.
                cand_proxy_sinr = compute_pmi_sinr_proxy_from_precoder(
                    q_vectors, cand_beams, noise_power, eps=rl_cfg.reward_sinr_eps
                )
                base_cand_reward = compute_rl_reward(
                    reward_mode=rl_cfg.reward_mode,
                    rate=cand_rate,
                    zf_rate=ref_rate,
                    actual_sinr=cand_actual_sinr,
                    zf_actual_sinr=ref_actual_sinr,
                    proxy_sinr=cand_proxy_sinr,
                    zf_proxy_sinr=ref_proxy_sinr,
                    eps=rl_cfg.reward_sinr_eps,
                )
                cand_reward = add_positive_rate_delta_bonus(
                    base_reward=base_cand_reward,
                    rate=cand_rate,
                    ref_rate=ref_rate,
                    bonus_lambda=rl_cfg.positive_rate_bonus_lambda,
                    bonus_power=rl_cfg.positive_rate_bonus_power,
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
                    best_candidate_pool = cand_candidate_pool
                    best_slnr_score_pool = cand_slnr_score_pool
                    best_selected_indices = cand_selected_indices
                    best_slnr_centers = cand_slnr_centers
                    best_diag = cand_diag
                    best_rate = cand_rate
                    best_actual_sinr = cand_actual_sinr
                    best_proxy_sinr = cand_proxy_sinr
                    best_reward = cand_reward

        else:
            # Exact domain-knowledge branch: use the cached PMI-SLNR precoder
            # without perturbation. This gives the hybrid policy a safe reuse
            # action and lets the learned branch specialize in slots where
            # perturbing SLNR is actually beneficial.
            best_score = 0.0
            best_candidate_index = 0
            best_beams = slnr_precoder_t.copy()
            best_x_sample = np.stack(
                [complex_unit_to_real(slnr_precoder_t[:, ku]) for ku in range(k)],
                axis=0,
            )
            best_signal = np.zeros(k, dtype=np.float64)
            best_leakage = np.zeros(k, dtype=np.float64)
            best_attempts = np.ones(k, dtype=np.float64)
            best_accept = np.ones(k, dtype=np.float64)
            best_emp_mean = best_x_sample.copy()
            m_count = max(1, rl_cfg.max_fb_resamples)
            best_candidate_pool = np.zeros((k, m_count, d), dtype=np.float64)
            best_slnr_score_pool = np.zeros((k, m_count), dtype=np.float64)
            best_selected_indices = np.zeros(k, dtype=np.int64)
            best_slnr_centers = best_x_sample.copy()
            for ku in range(k):
                best_candidate_pool[ku, 0] = best_x_sample[ku]
                best_signal[ku] = raw_quadratic_score(best_x_sample[ku], signal_mats[ku])
                best_leakage[ku] = raw_quadratic_score(best_x_sample[ku], leakage_mats[ku])
                ell = best_leakage[ku]
                g = best_signal[ku]
                best_slnr_score_pool[ku, 0] = rl_cfg.signal_gamma * g / max(ell + slnr_noise_power, rl_cfg.leakage_norm_eps)
            best_diag = {
                "entropy": np.zeros(k, dtype=np.float64),
                "entropy_norm": np.zeros(k, dtype=np.float64),
                "effective_candidates": np.ones(k, dtype=np.float64),
                "selected_policy_score": np.zeros(k, dtype=np.float64),
                "best_policy_score": np.zeros(k, dtype=np.float64),
                "mean_policy_score": np.zeros(k, dtype=np.float64),
                "selected_minus_mean_policy_score": np.zeros(k, dtype=np.float64),
                "phase_score_mean": np.zeros(k, dtype=np.float64),
                "phase_score_std": np.zeros(k, dtype=np.float64),
                "slnr_score_mean": np.zeros(k, dtype=np.float64),
                "slnr_score_std": np.zeros(k, dtype=np.float64),
                "policy_score_std": np.zeros(k, dtype=np.float64),
                "selected_source": np.full(k, 2.0, dtype=np.float64),
            }
            best_rate = slnr_rate
            best_actual_sinr = slnr_actual_sinr
            best_proxy_sinr = slnr_proxy_sinr
            base_best_reward = compute_rl_reward(
                reward_mode=rl_cfg.reward_mode,
                rate=best_rate,
                zf_rate=ref_rate,
                actual_sinr=best_actual_sinr,
                zf_actual_sinr=ref_actual_sinr,
                proxy_sinr=best_proxy_sinr,
                zf_proxy_sinr=ref_proxy_sinr,
                eps=rl_cfg.reward_sinr_eps,
            )
            best_reward = add_positive_rate_delta_bonus(
                base_reward=base_best_reward,
                rate=best_rate,
                ref_rate=ref_rate,
                bonus_lambda=rl_cfg.positive_rate_bonus_lambda,
                bonus_power=rl_cfg.positive_rate_bonus_power,
            )

        assert best_beams is not None
        beams = best_beams
        x_sample = best_x_sample
        signal_trace[t] = best_signal
        leakage_trace[t] = best_leakage
        fb_attempts_trace[t] = best_attempts
        fb_accept_trace[t] = best_accept
        emp_mean_sample = best_emp_mean
        candidate_pool_sample = best_candidate_pool
        slnr_score_pool_sample = best_slnr_score_pool
        selected_indices_sample = best_selected_indices
        slnr_centers_sample = best_slnr_centers
        diag_sample = best_diag

        # Cheap oracle diagnostic: if exact anchors are enabled, candidate index 0
        # is the cached PMI-SLNR beam for every UE. This should match the cached
        # SLNR baseline throughput almost exactly; otherwise the candidate pool
        # is not reproducing the baseline.
        if rl_cfg.include_exact_anchor_candidates and candidate_pool_sample is not None:
            oracle_slnr_beams = np.zeros((cfg.num_tx_antennas, k), dtype=np.complex128)
            for ku in range(k):
                oracle_slnr_beams[:, ku] = real_to_complex_beam(
                    candidate_pool_sample[ku, 0], cfg.total_tx_power, k
                )
            oracle_slnr_anchor_rate_trace[t], _ = compute_slot_sum_rate(
                channels[t], oracle_slnr_beams, noise_power
            )

        # Expensive optional diagnostic: score all M^K full-precoder combinations
        # from the per-UE candidate pools and record the best actual rate. For
        # K=2, M=32 this is 1024 rate evaluations per slot, so keep it off for
        # long runs unless explicitly requested.
        if rl_cfg.enable_oracle_pool_diagnostics and candidate_pool_sample is not None:
            from itertools import product
            m_count = candidate_pool_sample.shape[1]
            best_pool_rate = -np.inf
            for combo in product(range(m_count), repeat=k):
                pool_beams = np.zeros((cfg.num_tx_antennas, k), dtype=np.complex128)
                for ku, ci in enumerate(combo):
                    pool_beams[:, ku] = real_to_complex_beam(
                        candidate_pool_sample[ku, ci], cfg.total_tx_power, k
                    )
                pool_rate, _ = compute_slot_sum_rate(channels[t], pool_beams, noise_power)
                if pool_rate > best_pool_rate:
                    best_pool_rate = pool_rate
            oracle_best_pool_rate_trace[t] = best_pool_rate
        candidate_entropy_trace[t] = diag_sample["entropy"]
        candidate_entropy_norm_trace[t] = diag_sample["entropy_norm"]
        candidate_effective_count_trace[t] = diag_sample["effective_candidates"]
        candidate_selected_score_trace[t] = diag_sample["selected_policy_score"]
        candidate_best_score_trace[t] = diag_sample["best_policy_score"]
        candidate_mean_score_trace[t] = diag_sample["mean_policy_score"]
        candidate_selected_minus_mean_score_trace[t] = diag_sample["selected_minus_mean_policy_score"]
        candidate_phase_score_mean_trace[t] = diag_sample["phase_score_mean"]
        candidate_phase_score_std_trace[t] = diag_sample["phase_score_std"]
        candidate_slnr_score_mean_trace[t] = diag_sample["slnr_score_mean"]
        candidate_slnr_score_std_trace[t] = diag_sample["slnr_score_std"]
        candidate_policy_score_std_trace[t] = diag_sample["policy_score_std"]
        candidate_selected_source_trace[t] = diag_sample["selected_source"]
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
        rate_delta_slnr_trace[t] = rate - slnr_rate
        proxy_sinr_trace[t] = proxy_sinr
        zf_proxy_sinr_trace[t] = zf_proxy_sinr
        proxy_sinr_ratio_trace[t] = proxy_sinr / np.maximum(zf_proxy_sinr, rl_cfg.reward_sinr_eps)
        actual_sinr_ratio_trace[t] = actual_sinr / np.maximum(zf_actual_sinr, rl_cfg.reward_sinr_eps)
        proxy_sinr_ratio_slnr_trace[t] = proxy_sinr / np.maximum(slnr_proxy_sinr, rl_cfg.reward_sinr_eps)
        actual_sinr_ratio_slnr_trace[t] = actual_sinr / np.maximum(slnr_actual_sinr, rl_cfg.reward_sinr_eps)
        advantage_trace[t] = advantage
        baseline_trace[t] = reward_baseline
        beat_zf_trace[t] = 1.0 if rate > zf_rate else 0.0
        beat_slnr_trace[t] = 1.0 if rate > slnr_rate else 0.0
        beam_similarity_trace[t] = beam_similarity(beams, zf_baseline["precoders"][t])

        batch_s.append(s)
        batch_candidate_pools.append(candidate_pool_sample)
        batch_slnr_score_pools.append(slnr_score_pool_sample)
        batch_selected_indices.append(selected_indices_sample)
        batch_slnr_centers.append(slnr_centers_sample)
        batch_q_real.append(q_real_sample)
        batch_branch_is_learned.append(1.0 if slot_from_learned else 0.0)
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
        "rate_delta_slnr": rate_delta_slnr_trace,
        "proxy_sinr": proxy_sinr_trace,
        "zf_proxy_sinr": zf_proxy_sinr_trace,
        "proxy_sinr_ratio": proxy_sinr_ratio_trace,
        "actual_sinr_ratio": actual_sinr_ratio_trace,
        "proxy_sinr_ratio_slnr": proxy_sinr_ratio_slnr_trace,
        "actual_sinr_ratio_slnr": actual_sinr_ratio_slnr_trace,
        "advantage": advantage_trace,
        "reward_baseline": baseline_trace,
        "kappa": kappa_trace,
        "beat_zf": beat_zf_trace,
        "beat_slnr": beat_slnr_trace,
        "oracle_slnr_anchor_rate": oracle_slnr_anchor_rate_trace,
        "oracle_best_pool_rate": oracle_best_pool_rate_trace,
        "beam_similarity_to_zf": beam_similarity_trace,
        "grad_norm": grad_norm_trace,
        "loss": loss_trace,
        "rl_loss": rl_loss_trace,
        "joint_pmi_aux_loss": joint_pmi_aux_loss_trace,
        "signal": signal_trace,
        "leakage": leakage_trace,
        "fb_attempts": fb_attempts_trace,
        "fb_accept": fb_accept_trace,
        "best_of_n_score": best_of_n_score_trace,
        "best_of_n_selected": best_of_n_selected_trace,
        "empirical_mean_norm": empirical_mean_norm_trace,
        "centered_update_norm": centered_update_norm_trace,
        "candidate_entropy": candidate_entropy_trace,
        "candidate_entropy_norm": candidate_entropy_norm_trace,
        "candidate_effective_count": candidate_effective_count_trace,
        "candidate_selected_score": candidate_selected_score_trace,
        "candidate_best_score": candidate_best_score_trace,
        "candidate_mean_score": candidate_mean_score_trace,
        "candidate_selected_minus_mean_score": candidate_selected_minus_mean_score_trace,
        "candidate_phase_score_mean": candidate_phase_score_mean_trace,
        "candidate_phase_score_std": candidate_phase_score_std_trace,
        "candidate_slnr_score_mean": candidate_slnr_score_mean_trace,
        "candidate_slnr_score_std": candidate_slnr_score_std_trace,
        "candidate_policy_score_std": candidate_policy_score_std_trace,
        "candidate_selected_source": candidate_selected_source_trace,
        "learned_policy_probability": learned_policy_probability_trace,
        "dk_policy_probability": dk_policy_probability_trace,
        "selected_learned_branch": selected_learned_branch_trace,
        "selected_dk_branch": selected_dk_branch_trace,
        "mixture_logit": mixture_logit_trace,
        "wesn_states": wesn_states,
    }


def moving_average(trace: np.ndarray, window_len: int) -> np.ndarray:
    if window_len <= 1 or window_len > trace.size:
        return trace.copy()
    kernel = np.ones(window_len, dtype=np.float64) / window_len
    return np.convolve(trace, kernel, mode="valid")


def save_plots(
    zf_throughput: np.ndarray,
    slnr_throughput: np.ndarray,
    random_vmf_throughput: np.ndarray,
    rl_results: dict[str, np.ndarray],
    output_dir: Path,
    window_len: int,
) -> None:
    rl_throughput = rl_results["throughput"]
    reward = rl_results["reward"]
    beat_zf = rl_results["beat_zf"]
    beat_slnr = rl_results.get("beat_slnr", None)
    sim_to_zf = rl_results["beam_similarity_to_zf"].mean(axis=1)
    grad_norm = rl_results["grad_norm"]
    proxy_sinr_ratio = rl_results.get("proxy_sinr_ratio", None)
    actual_sinr_ratio = rl_results.get("actual_sinr_ratio", None)
    signal = rl_results.get("signal", None)
    leakage = rl_results.get("leakage", None)
    fb_attempts = rl_results.get("fb_attempts", None)
    fb_accept = rl_results.get("fb_accept", None)
    candidate_entropy_norm = rl_results.get("candidate_entropy_norm", None)
    candidate_effective_count = rl_results.get("candidate_effective_count", None)
    candidate_selected_minus_mean_score = rl_results.get("candidate_selected_minus_mean_score", None)
    learned_policy_probability = rl_results.get("learned_policy_probability", None)
    dk_policy_probability = rl_results.get("dk_policy_probability", None)
    selected_learned_branch = rl_results.get("selected_learned_branch", None)

    zf_avg = moving_average(zf_throughput, window_len)
    slnr_avg = moving_average(slnr_throughput, window_len)
    random_vmf_avg = moving_average(random_vmf_throughput, window_len)
    rl_avg = moving_average(rl_throughput, window_len)
    reward_avg = moving_average(reward, window_len)
    beat_zf_avg = moving_average(beat_zf, window_len)
    beat_slnr_avg = moving_average(beat_slnr, window_len) if beat_slnr is not None else None
    sim_avg = moving_average(sim_to_zf, window_len)
    grad_norm_avg = moving_average(grad_norm, window_len)
    proxy_sinr_ratio_avg = moving_average(proxy_sinr_ratio.mean(axis=1), window_len) if proxy_sinr_ratio is not None else None
    actual_sinr_ratio_avg = moving_average(actual_sinr_ratio.mean(axis=1), window_len) if actual_sinr_ratio is not None else None
    signal_avg = moving_average(signal.mean(axis=1), window_len) if signal is not None else None
    leakage_avg = moving_average(leakage.mean(axis=1), window_len) if leakage is not None else None
    fb_attempts_avg = moving_average(fb_attempts.mean(axis=1), window_len) if fb_attempts is not None else None
    fb_accept_avg = moving_average(fb_accept.mean(axis=1), window_len) if fb_accept is not None else None
    candidate_entropy_norm_avg = moving_average(candidate_entropy_norm.mean(axis=1), window_len) if candidate_entropy_norm is not None else None
    candidate_effective_count_avg = moving_average(candidate_effective_count.mean(axis=1), window_len) if candidate_effective_count is not None else None
    candidate_selected_minus_mean_score_avg = moving_average(candidate_selected_minus_mean_score.mean(axis=1), window_len) if candidate_selected_minus_mean_score is not None else None
    learned_policy_probability_avg = moving_average(learned_policy_probability, window_len) if learned_policy_probability is not None else None
    dk_policy_probability_avg = moving_average(dk_policy_probability, window_len) if dk_policy_probability is not None else None
    selected_learned_branch_avg = moving_average(selected_learned_branch, window_len) if selected_learned_branch is not None else None

    x_tput = np.arange(1, zf_avg.size + 1)
    x_reward = np.arange(1, reward_avg.size + 1)

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x_tput, zf_avg, lw=1.5, label="ZF baseline")
    ax1.plot(x_tput, slnr_avg, lw=1.5, label="PMI-SLNR baseline")
    # ax1.plot(x_tput, random_vmf_avg, lw=1.5, label="Random vMF baseline")
    ax1.plot(x_tput, rl_avg, lw=1.5, label="WESN")
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
    ax2.set_ylabel("Reward vs selected reference")
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

    if beat_slnr_avg is not None:
        fig3b, ax3b = plt.subplots(figsize=(8, 4.5))
        ax3b.plot(np.arange(1, beat_slnr_avg.size + 1), beat_slnr_avg, lw=1.5)
        ax3b.set_title("Fraction of Slots Where WESN-FB Raw-SLNR RL Beats SLNR")
        ax3b.set_xlabel("Slot index")
        ax3b.set_ylabel("Moving-average fraction")
        ax3b.set_ylim(0.0, 1.0)
        ax3b.grid(True, alpha=0.35)
        fig3b.tight_layout()
        fig3b.savefig(output_dir / "esn_vmf_rl_fraction_beats_slnr.png", dpi=150)
        plt.close(fig3b)

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

    if candidate_entropy_norm_avg is not None:
        fig9, ax9 = plt.subplots(figsize=(8, 4.5))
        ax9.plot(np.arange(1, candidate_entropy_norm_avg.size + 1), candidate_entropy_norm_avg, lw=1.5)
        ax9.set_title("Candidate Softmax Normalized Entropy")
        ax9.set_xlabel("Slot index")
        ax9.set_ylabel("Entropy / log(num candidates)")
        ax9.set_ylim(0.0, 1.05)
        ax9.grid(True, alpha=0.35)
        fig9.tight_layout()
        fig9.savefig(output_dir / "phase_inv_candidate_entropy_norm.png", dpi=150)
        plt.close(fig9)

    if candidate_effective_count_avg is not None:
        fig10, ax10 = plt.subplots(figsize=(8, 4.5))
        ax10.plot(np.arange(1, candidate_effective_count_avg.size + 1), candidate_effective_count_avg, lw=1.5)
        ax10.set_title("Candidate Softmax Effective Number of Candidates")
        ax10.set_xlabel("Slot index")
        ax10.set_ylabel("1 / sum_i p_i^2")
        ax10.grid(True, alpha=0.35)
        fig10.tight_layout()
        fig10.savefig(output_dir / "phase_inv_candidate_effective_count.png", dpi=150)
        plt.close(fig10)

    if candidate_selected_minus_mean_score_avg is not None:
        fig11, ax11 = plt.subplots(figsize=(8, 4.5))
        ax11.plot(np.arange(1, candidate_selected_minus_mean_score_avg.size + 1), candidate_selected_minus_mean_score_avg, lw=1.5)
        ax11.set_title("Selected Candidate Score Above Pool Mean")
        ax11.set_xlabel("Slot index")
        ax11.set_ylabel("Selected score - mean pool score")
        ax11.grid(True, alpha=0.35)
        fig11.tight_layout()
        fig11.savefig(output_dir / "phase_inv_candidate_selected_minus_mean_score.png", dpi=150)
        plt.close(fig11)



    if learned_policy_probability_avg is not None and dk_policy_probability_avg is not None:
        fig12, ax12 = plt.subplots(figsize=(8, 4.5))
        ax12.plot(np.arange(1, learned_policy_probability_avg.size + 1), learned_policy_probability_avg, lw=1.5, label="Learned SLNR-perturbation branch")
        ax12.plot(np.arange(1, dk_policy_probability_avg.size + 1), dk_policy_probability_avg, lw=1.5, label="Exact SLNR DK branch")
        ax12.set_title("Hybrid Mixture Branch Probabilities Across Time")
        ax12.set_xlabel("Slot index")
        ax12.set_ylabel("Probability")
        ax12.set_ylim(0.0, 1.0)
        ax12.grid(True, alpha=0.35)
        ax12.legend(loc="best")
        fig12.tight_layout()
        fig12.savefig(output_dir / "hybrid_mixture_branch_probabilities.png", dpi=150)
        plt.close(fig12)

    if selected_learned_branch_avg is not None:
        fig13, ax13 = plt.subplots(figsize=(8, 4.5))
        ax13.plot(np.arange(1, selected_learned_branch_avg.size + 1), selected_learned_branch_avg, lw=1.5)
        ax13.set_title("Fraction of Executed Slots from Learned SLNR-Perturbation Branch")
        ax13.set_xlabel("Slot index")
        ax13.set_ylabel("Moving-average fraction")
        ax13.set_ylim(0.0, 1.0)
        ax13.grid(True, alpha=0.35)
        fig13.tight_layout()
        fig13.savefig(output_dir / "hybrid_mixture_executed_learned_branch.png", dpi=150)
        plt.close(fig13)

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



def load_or_run_slnr_baseline(
    cfg: SimConfig,
    channels: np.ndarray,
    cache_dir: Path,
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """Load PMI-only SLNR baseline from cache if compatible; otherwise compute it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "slnr_baseline_meta.json"
    metadata = sim_cache_metadata(cfg) | {"baseline": "pmi_only_slnr"}

    paths = {
        "throughput": cache_dir / "slnr_throughput_trace.npy",
        "sinr": cache_dir / "slnr_sinr_trace.npy",
        "precoders": cache_dir / "slnr_precoders_trace.npy",
    }

    can_load = (
        not force_recompute
        and _metadata_matches(meta_path, metadata)
        and all(path.exists() for path in paths.values())
    )
    if can_load:
        print(f"Loading cached SLNR baseline from {cache_dir}")
        return {name: np.load(path) for name, path in paths.items()}

    print("Cached SLNR baseline not found or incompatible; recomputing.")
    slnr_results = run_slnr_baseline(cfg, channels)
    for name, path in paths.items():
        np.save(path, slnr_results[name])
    _write_metadata(meta_path, metadata)
    return slnr_results


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
    parser = argparse.ArgumentParser(description="Simple MU-MIMO ZF baseline + batched WESN phase-invariant PMI raw-SLNR RL")
    parser.add_argument("--num-slots", type=int, default=50000)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rl_precoder_design_per_UE_SLNR_perturbation_with_mixture"))
    parser.add_argument("--window-len", type=int, default=5000)
    parser.add_argument("--baseline-cache-dir", type=Path, default=None)
    parser.add_argument("--force-recompute-baselines", action="store_true")

    # WESN knobs. The reservoir update is the same as the vanilla ESN; the
    # readout is augmented with a window of recent PMI inputs as skip connections.
    # W_in and W_res are fixed; only W_out is learned.
    parser.add_argument("--reservoir-size", type=int, default=128)
    parser.add_argument("--spectral-radius", type=float, default=0.8)
    parser.add_argument("--input-scale", type=float, default=0.15)
    parser.add_argument(
        "--skip-window-length",
        type=int,
        default=4,
        help="Number of recent PMI feature vectors included in the WESN skip/readout connection. Use 1 for the one-step skip connection.",
    )

    # RL/training knobs.
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-out", type=float, default=3e-2)
    parser.add_argument("--fixed-kappa", type=float, default=3.0)
    parser.add_argument("--reward-baseline-beta", type=float, default=0.99)
    parser.add_argument("--advantage-clip", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--init-scale-out", type=float, default=1e-2)
    parser.add_argument("--leakage-lambda", type=float, default=1.0, help="Deprecated/unused: SLNR-ratio denominator now uses normalized noise power.")
    parser.add_argument("--signal-gamma", type=float, default=1)
    parser.add_argument("--max-fb-resamples", type=int, default=32)
    parser.add_argument("--leakage-norm-eps", type=float, default=1e-12)
    parser.add_argument("--signal-norm-eps", type=float, default=1e-12)
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="rate_log_ratio",
        choices=[
            "throughput_delta",
            "normalized_throughput_delta",
            "actual_sinr_log_ratio",
            "rate_log_ratio",
            "sinr_proxy_log_ratio",
        ],
        help="Scalar reward used for REINFORCE updates.",
    )
    parser.add_argument("--reward-sinr-eps", type=float, default=1e-12)
    parser.add_argument(
        "--reward-reference",
        type=str,
        default="slnr",
        choices=["zf", "slnr"],
        help="Reference baseline used inside the RL reward. Use slnr when trying to beat the SLNR baseline.",
    )
    parser.add_argument("--best-of-n", type=int, default=1, help="Must be 1 for exact on-policy candidate-based training. Use --max-fb-resamples for per-UE candidate directions.")
    parser.add_argument("--candidate-temperature", type=float, default=0.8, help="Softmax temperature for candidate selection; smaller is stricter.")
    parser.add_argument("--no-normalize-candidate-scores", action="store_true", help="Disable per-pool z-score normalization of phase and SLNR candidate scores.")
    parser.add_argument("--uniform-candidate-frac", type=float, default=0.10, help="Fraction of candidates sampled uniformly.")
    parser.add_argument("--pmi-candidate-frac", type=float, default=0.35, help="Fraction of candidates sampled around the PMI direction.")
    parser.add_argument("--slnr-candidate-frac", type=float, default=0.35, help="Fraction of candidates sampled around the PMI-SLNR heuristic direction.")
    parser.add_argument("--wesn-candidate-frac", type=float, default=0.20, help="Fraction of candidates sampled around the current WESN policy mean. This improves learned proposal diversity but uses a stop-gradient proposal approximation in the training loss.")
    parser.add_argument("--candidate-proposal-kappa", type=float, default=50.0, help="Concentration for PMI/SLNR/WESN vMF proposal candidates.")
    parser.add_argument("--slnr-residual-scale", type=float, default=0.15, help="Scale alpha in normalize(v_SLNR + alpha * Delta(mu)); this only affects the WESN selector center, not the structured candidate locations.")
    parser.add_argument(
        "--structured-leakage-alpha-grid",
        type=str,
        default="0,0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.80,1.00",
        help="Comma-separated nonnegative alpha values for structured candidates v(alpha)=normalize(v_SLNR + alpha*u_leak). Negative values are ignored in this RL version.",
    )
    parser.add_argument("--alpha-slnr-prior-weight", type=float, default=0.0, help="Optional fixed PMI-SLNR prior weight added to WESN direct-alpha logits. Default 0 means WESN learns alpha logits directly.")
    parser.add_argument("--no-exact-anchor-candidates", action="store_true", help="Disable deterministic exact SLNR / SLNR-residual / PMI candidates in the candidate pool.")
    parser.add_argument("--joint-pmi-aux-loss-weight", type=float, default=0.0, help="Weight of the joint PMI-only full-precoder auxiliary ranking loss. Use 0 to disable.")
    parser.add_argument("--joint-pmi-aux-temperature", type=float, default=0.5, help="Softmax temperature used to convert joint PMI proxy rates into auxiliary target weights.")
    parser.add_argument("--num-joint-pmi-aux-candidates", type=int, default=32, help="Number of full-precoder candidate combinations sampled per slot for the joint PMI auxiliary loss.")
    parser.add_argument("--enable-oracle-pool-diagnostics", action="store_true", help="Expensively score all M^K full-precoder combinations from the candidate pools and log the best actual rate.")
    parser.add_argument("--initial-learned-policy-probability", type=float, default=0.8, help="Initial reuse probability for the learned SLNR-perturbation branch. The exact SLNR DK branch starts with probability 1-p.")
    parser.add_argument("--lr-mixture", type=float, default=3e-2, help="Learning rate for the trainable mixture/reuse probability logit.")
    parser.add_argument("--positive-rate-bonus-lambda", type=float, default=2.0, help="Scale lambda for the positive rate-delta bonus: lambda * max(rate - reference_rate, 0)^p.")
    parser.add_argument("--positive-rate-bonus-power", type=float, default=0.5, help="Exponent p for the positive rate-delta bonus. Default 0.5 amplifies small positive deltas.")
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
        skip_window_length=args.skip_window_length,
        leakage_lambda=args.leakage_lambda,
        signal_gamma=args.signal_gamma,
        max_fb_resamples=args.max_fb_resamples,
        leakage_norm_eps=args.leakage_norm_eps,
        signal_norm_eps=args.signal_norm_eps,
        reward_mode=args.reward_mode,
        reward_reference=args.reward_reference,
        reward_sinr_eps=args.reward_sinr_eps,
        best_of_n=args.best_of_n,
        candidate_temperature=args.candidate_temperature,
        normalize_candidate_scores=not args.no_normalize_candidate_scores,
        uniform_candidate_frac=args.uniform_candidate_frac,
        pmi_candidate_frac=args.pmi_candidate_frac,
        slnr_candidate_frac=args.slnr_candidate_frac,
        wesn_candidate_frac=args.wesn_candidate_frac,
        candidate_proposal_kappa=args.candidate_proposal_kappa,
        slnr_residual_scale=args.slnr_residual_scale,
        include_exact_anchor_candidates=not args.no_exact_anchor_candidates,
        structured_leakage_alpha_grid=parse_nonnegative_alpha_grid(args.structured_leakage_alpha_grid),
        alpha_slnr_prior_weight=args.alpha_slnr_prior_weight,
        joint_pmi_aux_loss_weight=args.joint_pmi_aux_loss_weight,
        enable_oracle_pool_diagnostics=args.enable_oracle_pool_diagnostics,
        joint_pmi_aux_temperature=args.joint_pmi_aux_temperature,
        num_joint_pmi_aux_candidates=args.num_joint_pmi_aux_candidates,
        initial_learned_policy_probability=args.initial_learned_policy_probability,
        lr_mixture=args.lr_mixture,
        positive_rate_bonus_lambda=args.positive_rate_bonus_lambda,
        positive_rate_bonus_power=args.positive_rate_bonus_power,
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
    slnr_results = load_or_run_slnr_baseline(
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
    rl_results = run_wesn_policy_rl(cfg, rl_cfg, channels, zf_results, slnr_results, rng_rl)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "zf_throughput_trace.npy", zf_results["throughput"])
    np.save(args.output_dir / "slnr_throughput_trace.npy", slnr_results["throughput"])
    np.save(args.output_dir / "slnr_sinr_trace.npy", slnr_results["sinr"])
    np.save(args.output_dir / "slnr_precoders_trace.npy", slnr_results["precoders"])
    np.save(args.output_dir / "random_vmf_baseline_throughput_trace.npy", random_vmf_results["throughput"])
    np.save(args.output_dir / "esn_vmf_rl_throughput_trace.npy", rl_results["throughput"])
    np.save(args.output_dir / "esn_vmf_rl_reward_trace.npy", rl_results["reward"])
    np.save(args.output_dir / "esn_vmf_rl_rate_delta_trace.npy", rl_results["rate_delta"])
    np.save(args.output_dir / "esn_vmf_rl_rate_delta_slnr_trace.npy", rl_results["rate_delta_slnr"])
    np.save(args.output_dir / "esn_fb_proxy_sinr_trace.npy", rl_results["proxy_sinr"])
    np.save(args.output_dir / "esn_fb_zf_proxy_sinr_trace.npy", rl_results["zf_proxy_sinr"])
    np.save(args.output_dir / "esn_fb_proxy_sinr_ratio_trace.npy", rl_results["proxy_sinr_ratio"])
    np.save(args.output_dir / "esn_fb_actual_sinr_ratio_trace.npy", rl_results["actual_sinr_ratio"])
    np.save(args.output_dir / "esn_fb_proxy_sinr_ratio_slnr_trace.npy", rl_results["proxy_sinr_ratio_slnr"])
    np.save(args.output_dir / "esn_fb_actual_sinr_ratio_slnr_trace.npy", rl_results["actual_sinr_ratio_slnr"])
    np.save(args.output_dir / "esn_vmf_rl_advantage_trace.npy", rl_results["advantage"])
    np.save(args.output_dir / "esn_vmf_rl_reward_baseline_trace.npy", rl_results["reward_baseline"])
    np.save(args.output_dir / "esn_vmf_rl_kappa_trace.npy", rl_results["kappa"])
    np.save(args.output_dir / "esn_vmf_rl_fraction_beats_zf_trace.npy", rl_results["beat_zf"])
    np.save(args.output_dir / "esn_vmf_rl_fraction_beats_slnr_trace.npy", rl_results["beat_slnr"])
    np.save(args.output_dir / "oracle_slnr_anchor_rate_trace.npy", rl_results["oracle_slnr_anchor_rate"])
    np.save(args.output_dir / "oracle_best_pool_rate_trace.npy", rl_results["oracle_best_pool_rate"])
    np.save(args.output_dir / "esn_vmf_rl_beam_similarity_to_zf_trace.npy", rl_results["beam_similarity_to_zf"])
    np.save(args.output_dir / "esn_vmf_rl_grad_norm_trace.npy", rl_results["grad_norm"])
    np.save(args.output_dir / "esn_vmf_rl_loss_trace.npy", rl_results["loss"])
    np.save(args.output_dir / "esn_vmf_rl_rl_loss_trace.npy", rl_results["rl_loss"])
    np.save(args.output_dir / "esn_vmf_rl_joint_pmi_aux_loss_trace.npy", rl_results["joint_pmi_aux_loss"])
    np.save(args.output_dir / "esn_fb_signal_trace.npy", rl_results["signal"])
    np.save(args.output_dir / "esn_fb_leakage_trace.npy", rl_results["leakage"])
    np.save(args.output_dir / "esn_fb_sampler_attempts_trace.npy", rl_results["fb_attempts"])
    np.save(args.output_dir / "esn_fb_sampler_accept_trace.npy", rl_results["fb_accept"])
    np.save(args.output_dir / "esn_fb_best_of_n_score_trace.npy", rl_results["best_of_n_score"])
    np.save(args.output_dir / "esn_fb_best_of_n_selected_trace.npy", rl_results["best_of_n_selected"])
    np.save(args.output_dir / "esn_fb_empirical_mean_norm_trace.npy", rl_results["empirical_mean_norm"])
    np.save(args.output_dir / "esn_fb_centered_update_norm_trace.npy", rl_results["centered_update_norm"])
    np.save(args.output_dir / "phase_inv_candidate_entropy_trace.npy", rl_results["candidate_entropy"])
    np.save(args.output_dir / "phase_inv_candidate_entropy_norm_trace.npy", rl_results["candidate_entropy_norm"])
    np.save(args.output_dir / "phase_inv_candidate_effective_count_trace.npy", rl_results["candidate_effective_count"])
    np.save(args.output_dir / "phase_inv_candidate_selected_score_trace.npy", rl_results["candidate_selected_score"])
    np.save(args.output_dir / "phase_inv_candidate_best_score_trace.npy", rl_results["candidate_best_score"])
    np.save(args.output_dir / "phase_inv_candidate_mean_score_trace.npy", rl_results["candidate_mean_score"])
    np.save(args.output_dir / "phase_inv_candidate_selected_minus_mean_score_trace.npy", rl_results["candidate_selected_minus_mean_score"])
    np.save(args.output_dir / "phase_inv_candidate_phase_score_mean_trace.npy", rl_results["candidate_phase_score_mean"])
    np.save(args.output_dir / "phase_inv_candidate_phase_score_std_trace.npy", rl_results["candidate_phase_score_std"])
    np.save(args.output_dir / "phase_inv_candidate_slnr_score_mean_trace.npy", rl_results["candidate_slnr_score_mean"])
    np.save(args.output_dir / "phase_inv_candidate_slnr_score_std_trace.npy", rl_results["candidate_slnr_score_std"])
    np.save(args.output_dir / "phase_inv_candidate_policy_score_std_trace.npy", rl_results["candidate_policy_score_std"])
    np.save(args.output_dir / "phase_inv_candidate_selected_source_trace.npy", rl_results["candidate_selected_source"])
    np.save(args.output_dir / "hybrid_mixture_learned_policy_probability_trace.npy", rl_results["learned_policy_probability"])
    np.save(args.output_dir / "hybrid_mixture_dk_policy_probability_trace.npy", rl_results["dk_policy_probability"])
    np.save(args.output_dir / "hybrid_mixture_selected_learned_branch_trace.npy", rl_results["selected_learned_branch"])
    np.save(args.output_dir / "hybrid_mixture_selected_dk_branch_trace.npy", rl_results["selected_dk_branch"])
    np.save(args.output_dir / "hybrid_mixture_logit_trace.npy", rl_results["mixture_logit"])
    np.save(args.output_dir / "wesn_states_trace.npy", rl_results["wesn_states"])

    save_plots(
        zf_throughput=zf_results["throughput"],
        slnr_throughput=slnr_results["throughput"],
        random_vmf_throughput=random_vmf_results["throughput"],
        rl_results=rl_results,
        output_dir=args.output_dir,
        window_len=args.window_len,
    )

    print("Simple WESN-FB PMI raw-SLNR RL precoder design run finished.")
    print(f"ZF average throughput         : {zf_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"PMI-SLNR baseline throughput  : {slnr_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Random vMF baseline throughput: {random_vmf_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"WESN-FB PMI raw-SLNR RL throughput  : {rl_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Reward mode                   : {rl_cfg.reward_mode}")
    print(f"Reward reference              : {rl_cfg.reward_reference}")
    print(f"Positive rate bonus           : lambda={rl_cfg.positive_rate_bonus_lambda:.4f}, p={rl_cfg.positive_rate_bonus_power:.4f}  (lambda * max(rate-reference,0)^p)")
    print(f"WESN-FB PMI raw-SLNR RL reward      : {rl_results['reward'].mean():.4f}")
    print(f"WESN-FB throughput delta vs ZF  : {rl_results['rate_delta'].mean():.4f}")
    print(f"WESN-FB throughput delta vs SLNR: {rl_results['rate_delta_slnr'].mean():.4f}")
    print(f"Mean proxy-SINR ratio to ZF   : {rl_results['proxy_sinr_ratio'].mean():.4f}")
    print(f"Mean actual SINR ratio to ZF  : {rl_results['actual_sinr_ratio'].mean():.4f}")
    print(f"Mean proxy-SINR ratio to SLNR : {rl_results['proxy_sinr_ratio_slnr'].mean():.4f}")
    print(f"Mean actual SINR ratio to SLNR: {rl_results['actual_sinr_ratio_slnr'].mean():.4f}")
    print(f"RL beats ZF fraction          : {rl_results['beat_zf'].mean():.4f}")
    print(f"RL beats SLNR fraction        : {rl_results['beat_slnr'].mean():.4f}")
    print(f"Mean beam similarity to ZF    : {rl_results['beam_similarity_to_zf'].mean():.4f}")
    print(f"Mean grad norm                : {rl_results['grad_norm'].mean():.4f}")
    print(f"Oracle exact candidate-SLNR throughput: {np.nanmean(rl_results['oracle_slnr_anchor_rate']):.4f}")
    print(f"True cached SLNR throughput          : {slnr_results['throughput'].mean():.4f}")
    if np.any(np.isfinite(rl_results['oracle_best_pool_rate'])):
        print(f"Oracle best candidate-pool throughput: {np.nanmean(rl_results['oracle_best_pool_rate']):.4f}")
    print(f"Mean raw PMI desired signal: {rl_results['signal'].mean():.4f}")
    print(f"Mean raw PMI leakage       : {rl_results['leakage'].mean():.4f}")
    print(f"Direct-alpha sampler       : yes (WESN outputs logits over structured leakage alpha candidates)")
    print(f"Hybrid mixture policy      : yes (exact SLNR DK branch + learned SLNR-perturbation branch)")
    print(f"Initial learned branch prob: {rl_cfg.initial_learned_policy_probability:.4f}")
    print(f"Mean learned branch prob   : {rl_results['learned_policy_probability'].mean():.4f}")
    print(f"Final learned branch prob  : {rl_results['learned_policy_probability'][-1]:.4f}")
    print(f"Mean DK SLNR branch prob   : {rl_results['dk_policy_probability'].mean():.4f}")
    print(f"Selected learned branch frac: {rl_results['selected_learned_branch'].mean():.4f}")
    print(f"Selected DK SLNR branch frac: {rl_results['selected_dk_branch'].mean():.4f}")
    print(f"Mean mixture logit         : {rl_results['mixture_logit'].mean():.4f}")
    print(f"Mixture logit LR           : {rl_cfg.lr_mixture:.4e}")
    print(f"Structured leakage search  : yes (positive one-direction leakage-reduction steps from SLNR)")
    print(f"Leakage alpha grid         : {rl_cfg.structured_leakage_alpha_grid}")
    print(f"WESN alpha policy          : direct categorical logits over alpha grid")
    print(f"Alpha SLNR prior weight    : {rl_cfg.alpha_slnr_prior_weight:.4f}")
    print(f"Exact anchor candidates    : {rl_cfg.include_exact_anchor_candidates}")
    print(f"Candidate update           : conditional log-softmax with stop-gradient residual candidate proposal)")
    print(f"Candidate temperature      : {rl_cfg.candidate_temperature:.4f}")
    print(f"Normalize candidate scores : {rl_cfg.normalize_candidate_scores}")
    print(f"Candidate proposal fractions [uniform, PMI, SLNR, WESN]: ignored for structured leakage search")
    print(f"Candidate proposal kappa   : ignored for structured leakage search")
    print(f"Joint PMI aux loss weight  : {rl_cfg.joint_pmi_aux_loss_weight:.4f}")
    print(f"Oracle pool diagnostics    : {rl_cfg.enable_oracle_pool_diagnostics}")
    print(f"Joint PMI aux temperature  : {rl_cfg.joint_pmi_aux_temperature:.4f}")
    print(f"Joint PMI aux candidates   : {rl_cfg.num_joint_pmi_aux_candidates}")
    print(f"Mean RL loss               : {rl_results['rl_loss'].mean():.4f}")
    print(f"Mean joint PMI auxiliary loss: {rl_results['joint_pmi_aux_loss'].mean():.4f}")
    print(f"Empirical mean correction  : no (not used for phase-invariant quadratic score)")
    print(f"Mean empirical mean norm   : {rl_results['empirical_mean_norm'].mean():.4f}")
    print(f"Mean centered update norm  : {rl_results['centered_update_norm'].mean():.4f}")
    print(f"SLNR score uses raw PMI-only actual SLNR: yes")
    print(f"Candidate sampler selection rate: {rl_results['fb_accept'].mean():.4f}")
    print(f"Mean selected per-UE candidate index: {rl_results['fb_attempts'].mean():.4f}")
    print(f"Mean candidate normalized entropy: {rl_results['candidate_entropy_norm'].mean():.4f}")
    print(f"Mean effective candidates    : {rl_results['candidate_effective_count'].mean():.4f}")
    print(f"Mean selected score - pool mean: {rl_results['candidate_selected_minus_mean_score'].mean():.4f}")
    print(f"Mean candidate alpha-logit std: {rl_results['candidate_phase_score_std'].mean():.4f}")
    print(f"Mean candidate SLNR score std : {rl_results['candidate_slnr_score_std'].mean():.4f}")
    print(f"Selected source fractions [uniform, PMI, SLNR, structured leakage]: {(rl_results['candidate_selected_source'] == 0).mean():.4f}, {(rl_results['candidate_selected_source'] == 1).mean():.4f}, {(rl_results['candidate_selected_source'] == 2).mean():.4f}, {(rl_results['candidate_selected_source'] == 3).mean():.4f}")
    print(f"WESN skip window length      : {rl_cfg.skip_window_length}")
    print(f"Best-of-N candidates          : {rl_cfg.best_of_n}")
    print(f"Mean selected candidate index : {rl_results['best_of_n_selected'].mean():.4f}")


if __name__ == "__main__":
    main()