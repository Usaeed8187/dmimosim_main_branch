from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)

from dmimo.mimo.quantized_CSI_feedback import quantized_CSI_feedback

@dataclass
class SimConfig:
    num_tx_antennas: int = 4
    num_users: int = 2
    num_rx_antennas_per_user: int = 2
    streams_per_user: int = 1
    num_slots: int = 2000
    snr_db: float = 10.0
    temporal_correlation: float = 0.95
    # Channel generation model. toy_ar1 keeps the original independent AR(1)
    # coefficients. sionna_umi/sionna_uma/sionna_rma use Sionna TR 38.901
    # geometry-based stochastic channels to obtain more realistic temporal
    # evolution, cross-antenna correlation, and topology-dependent UE links.
    channel_model: str = "toy_ar1"
    carrier_frequency_hz: float = 3.5e9
    subcarrier_spacing_hz: float = 30e3
    slot_duration_s: float = 1e-3
    ue_speed_kmh: float = 30.0
    bs_height_m: float = 25.0
    ue_height_m: float = 1.5
    cell_radius_m: float = 100.0
    min_ue_distance_m: float = 10.0
    sionna_scenario: str = "umi"
    sionna_o2i_model: str = "low"
    sionna_enable_pathloss: bool = False
    sionna_enable_shadow_fading: bool = False
    seed: int = 7
    total_tx_power: float = 1.0
    # PMI feedback source used by the toy RL testbench.
    # right_singular_vectors keeps the original unquantized-SVD behavior.
    # type_ii calls dmimo.mimo.quantized_CSI_feedback to reconstruct Type-II PMI.
    pmi_feedback_mode: str = "right_singular_vectors"
    type_ii_feedback_architecture: str = "dMIMO_phase2_type_II_CB2"
    type_ii_nfft: int = 512
    type_ii_num_ofdm_symbols: int = 14


@dataclass
class RLConfig:
    lr_out: float = 3e-2
    fixed_kappa: float = 3.0
    reward_baseline_beta: float = 0.99  # Deprecated/ignored: no moving-average advantage baseline is used.
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
    reference_precoder: str = "slnr"
    reward_sinr_eps: float = 1e-12
    # Diagnostic only for now: maintain a reference-like running throughput
    # baseline using slots where the executed learned precoder is close to the
    # internally computed reference precoder. This trace is plotted but is not
    # used in the RL reward yet.
    reference_like_baseline_alpha_max: float = 0.01
    reference_like_baseline_sigma_d: float = 0.15
    reference_like_baseline_eps: float = 1e-12
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
    # reference-precoder perturbation candidate policy. Both this reuse probability and W_out
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
    # Direct categorical PMI-span policy. For K=2, each UE candidate is
    #   v_k(theta,phi)=cos(theta) q_k + exp(j phi) sin(theta) b_k,
    # where b_k is the other UE's PMI vector after removing the component
    # parallel to q_k. W_out outputs logits over this theta/phi grid.
    alpha_slnr_prior_weight: float = 0.0
    pmi_span_num_theta: int = 8
    pmi_span_num_phi: int = 8
    pmi_span_theta_max: float = float(np.pi / 2.0)
    # Continuous joint Gaussian PMI-span policy. The WESN outputs the mean of
    # a 2K-dimensional raw Gaussian action. For K=2, the raw action has four
    # coordinates [u_theta0, u_phi0, u_theta1, u_phi1]. These raw coordinates
    # are mapped to bounded PMI-span angles using sigmoids:
    #   theta_k = theta_max * sigmoid(u_theta,k)
    #   phi_k   = 2*pi      * sigmoid(u_phi,k).
    # Exploration standard deviations are fixed for now.
    gaussian_raw_std_theta: float = 0.50
    gaussian_raw_std_phi: float = 0.75
    # Structured angle-step policy.  The Gaussian action is now a raw 4D
    # step-size variable u_alpha, not a raw angle residual.  The executed
    # angles are
    #   x = x_SLNR + (alpha_max * tanh(u_alpha)) ⊙ d_proxy,
    # where d_proxy is the normalized PMI-only proxy-rate gradient evaluated
    # at the SLNR angle center.
    alpha_step_max_theta: float = 0.25
    alpha_step_max_phi: float = 0.75
    # Discrete diagonal alpha-step policy.  Each of the 4 coordinates chooses
    # one multiplier from this grid.  The executed angle step is
    #   delta_x_i = alpha_step_max_i * alpha_level_i * d_proxy_i.
    alpha_level_grid: tuple[float, ...] = (-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0)
    # Optional fixed prior favoring the zero-alpha/SLNR action at initialization.
    # W_out still learns additive logits over the same discrete levels.
    alpha_zero_logit_bias: float = 2.0
    # Multi-iteration sequential refinement policy.  This is a short inner MDP
    # inside each slot: starting from the reference beam, the policy repeatedly
    # chooses stop or a PMI-informed direction/alpha move, then receives one
    # terminal reward from the final precoder.
    multi_iter_max_steps: int = 3
    multi_iter_step_penalty: float = 0.0
    multi_iter_include_stop_action: bool = True
    # Deprecated constant initial physical-angle center, retained only so older
    # command lines do not break. The active policy uses a slot-dependent
    # PMI-SLNR angle center instead of a constant bias.
    gaussian_init_theta: float = 0.40
    gaussian_init_phi: float = float(np.pi)


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


def normalize_pmi_q_vectors(q_vectors: np.ndarray) -> np.ndarray:
    """Normalize a [num_users, num_tx_antennas] PMI direction matrix."""
    q_vectors = np.asarray(q_vectors, dtype=np.complex128)
    norms = np.maximum(np.linalg.norm(q_vectors, axis=1, keepdims=True), 1e-12)
    return q_vectors / norms


def build_zf_precoder_from_q_vectors(q_vectors: np.ndarray, total_tx_power: float) -> np.ndarray:
    """Build a ZF precoder from one PMI direction per UE."""
    q_vectors = normalize_pmi_q_vectors(q_vectors)
    z_matrix = q_vectors.conj()
    return normalize_columns_equal_power(np.linalg.pinv(z_matrix), total_tx_power)

def build_zf_precoder_from_pmi(
    full_vk_list: list[np.ndarray], streams_per_user: int, total_tx_power: float
) -> np.ndarray:
    if streams_per_user != 1:
        raise ValueError("PMI-only ZF baseline currently assumes one stream per UE.")
    q_vectors = np.stack([vk[:, 0] for vk in full_vk_list], axis=0)
    return build_zf_precoder_from_q_vectors(q_vectors, total_tx_power)


def build_slnr_precoder_from_q_vectors(
    q_vectors: np.ndarray,
    total_tx_power: float,
    noise_power: float,
) -> np.ndarray:
    """Build a PMI-only SLNR precoder from one PMI direction per UE.

    For each UE k, only q_k is used. The beam direction is obtained from

        max_v |q_k^H v|^2 / (sum_{j != k} |q_j^H v|^2 + sigma_eff^2),

    where sigma_eff^2 is the noise power normalized by the equal per-user
    transmit power. This keeps the baseline comparable between unquantized
    right-singular-vector PMI and quantized Type-II PMI.
    """
    q_vectors = normalize_pmi_q_vectors(q_vectors)
    num_users, num_tx_antennas = q_vectors.shape
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

def build_slnr_precoder_from_pmi(
    full_vk_list: list[np.ndarray],
    streams_per_user: int,
    total_tx_power: float,
    noise_power: float,
) -> np.ndarray:
    if streams_per_user != 1:
        raise ValueError("PMI-only SLNR baseline currently assumes one stream per UE.")
    q_vectors = np.stack([vk[:, 0] for vk in full_vk_list], axis=0)
    return build_slnr_precoder_from_q_vectors(q_vectors, total_tx_power, noise_power)

def pmi_features_from_q_vectors(q_vectors: np.ndarray) -> np.ndarray:
    q_vectors = normalize_pmi_q_vectors(q_vectors)
    return np.concatenate([np.real(q_vectors).reshape(-1), np.imag(q_vectors).reshape(-1)]).astype(np.float64)


def validate_type_ii_feedback_config(cfg: SimConfig) -> None:
    """Validate the toy-to-Type-II adapter against quantized_CSI_feedback assumptions."""
    if cfg.num_tx_antennas != 4:
        raise ValueError("Type-II PMI feedback mode currently requires cfg.num_tx_antennas == 4.")
    if cfg.num_rx_antennas_per_user != 2:
        raise ValueError("Type-II PMI feedback mode currently requires cfg.num_rx_antennas_per_user == 2.")
    if cfg.streams_per_user != 1:
        raise ValueError("Type-II PMI feedback mode currently supports one stream per UE.")
    if cfg.type_ii_feedback_architecture != "dMIMO_phase2_type_II_CB2":
        raise ValueError("The toy Type-II adapter currently targets dMIMO_phase2_type_II_CB2 only.")
    if cfg.type_ii_nfft != 512:
        raise ValueError("quantized_CSI_feedback currently hardcodes nfft=512 for Type-II feedback.")
    if cfg.type_ii_num_ofdm_symbols <= 0:
        raise ValueError("type_ii_num_ofdm_symbols must be positive.")


def make_type_ii_feedback_quantizer(cfg: SimConfig):
    """Construct the production Type-II PMI feedback layer used by the MU-MIMO pipeline.

    dMIMO_phase2_type_II_CB2 assumes two local BS receive antenna groups before
    the scheduled UE receive antenna groups. The toy test only has UE channels,
    so the adapter pads those two pseudo groups and therefore asks the quantizer
    for cfg.num_users + 2 rank-one PMI streams, then drops the two pseudo streams.
    """
    validate_type_ii_feedback_config(cfg)

    return quantized_CSI_feedback(
        method="5G",
        codebook_selection_method=None,
        num_tx_streams=cfg.num_users + 2,
        architecture=cfg.type_ii_feedback_architecture,
        rbs_per_subband=4,
        snrdb=cfg.snr_db,
    )


def toy_channels_to_type_ii_csi(user_channels: np.ndarray, cfg: SimConfig):
    """Map toy flat UE channels to quantized_CSI_feedback's rank-7 CSI tensor.

    Input user_channels has shape [K, Nrx_per_UE, Nt]. The returned tensor has
    shape [1, 1, 4 + K*Nrx_per_UE, 1, Nt, num_ofdm_symbols, nfft], matching the
    MU-MIMO pipeline's [batch, num_rx, rx_ant, num_tx, tx_ant, symbol, subcarrier]
    convention after scheduled-UE gathering.
    """
    validate_type_ii_feedback_config(cfg)
    import tensorflow as tf

    user_channels = np.asarray(user_channels, dtype=np.complex64)
    expected_shape = (cfg.num_users, cfg.num_rx_antennas_per_user, cfg.num_tx_antennas)
    if user_channels.shape != expected_shape:
        raise ValueError(f"Expected user_channels shape {expected_shape}, got {user_channels.shape}.")

    rx_offset = 4
    total_rx_antennas = rx_offset + cfg.num_users * cfg.num_rx_antennas_per_user
    csi = np.zeros(
        (1, 1, total_rx_antennas, 1, cfg.num_tx_antennas, cfg.type_ii_num_ofdm_symbols, cfg.type_ii_nfft),
        dtype=np.complex64,
    )

    # Non-zero pseudo BS receive groups keep the production Type-II CB2 path
    # numerically well-defined. Their resulting PMI streams are discarded below.
    pseudo_bs = np.zeros((rx_offset, cfg.num_tx_antennas), dtype=np.complex64)
    pseudo_bs[: cfg.num_tx_antennas, : cfg.num_tx_antennas] = np.eye(cfg.num_tx_antennas, dtype=np.complex64)
    csi[0, 0, :rx_offset, 0, :, :, :] = pseudo_bs[:, :, None, None]

    for k in range(cfg.num_users):
        start = rx_offset + k * cfg.num_rx_antennas_per_user
        stop = start + cfg.num_rx_antennas_per_user
        csi[0, 0, start:stop, 0, :, :, :] = user_channels[k, :, :, None, None]

    return tf.convert_to_tensor(csi, dtype=tf.complex64)


def type_ii_pmi_from_channels(
    user_channels: np.ndarray,
    cfg: SimConfig,
    quantizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Type-II quantized PMI features and q-vectors for one toy slot."""
    csi = toy_channels_to_type_ii_csi(user_channels, cfg)
    h_quant, _feedback_bits = quantizer(csi, return_feedback_bits=True)
    h_quant_np = np.asarray(h_quant.numpy())

    # h_quant: [1, 1, total_streams, 1, Nt, Nsym, Nfft]. Since the toy channel
    # is frequency-flat in the adapter, one representative RE is enough.
    all_streams = h_quant_np[0, 0, :, 0, :, 0, 0]
    real_ue_streams = all_streams[2 : 2 + cfg.num_users, :]
    q_vectors = normalize_pmi_q_vectors(real_ue_streams)
    return pmi_features_from_q_vectors(q_vectors), q_vectors


def build_pmi_feedback_trace(cfg: SimConfig, channels: np.ndarray) -> dict[str, np.ndarray]:
    """Build the per-slot PMI trace used consistently by ZF, SLNR, and RL."""
    mode = cfg.pmi_feedback_mode
    features: list[np.ndarray] = []
    q_vectors: list[np.ndarray] = []

    if mode == "right_singular_vectors":
        for t in range(cfg.num_slots):
            feat, vk_list = pmi_features_from_channels(channels[t])
            features.append(feat)
            q_vectors.append(normalize_pmi_q_vectors(np.stack([vk[:, 0] for vk in vk_list], axis=0)))
    elif mode == "type_ii":
        quantizer = make_type_ii_feedback_quantizer(cfg)
        for t in range(cfg.num_slots):
            print(f"Type-II PMI Slot {t + 1} / {cfg.num_slots}", end="\r")
            feat, q = type_ii_pmi_from_channels(channels[t], cfg, quantizer)
            features.append(feat)
            q_vectors.append(q)
        print()
    else:
        raise ValueError(f"Unsupported pmi_feedback_mode={mode!r}.")

    return {
        "features": np.stack(features, axis=0),
        "q_vectors": np.stack(q_vectors, axis=0),
    }

def compute_slot_sum_rate(
    user_channels: np.ndarray,
    precoder: np.ndarray,
    noise_power: float,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Compute sum-rate with a local effective-channel LMMSE receiver at each UE.

    For UE k, define the locally visible DMRS-style effective channels

        g_{k,j} = H_k p_j,   j = 1,...,K.

    This function assumes UE k has a perfect estimate of its own effective
    post-precoding matrix

        G_k = H_k P = [g_{k,1}, ..., g_{k,K}],

    but it does not assume UE k knows any other UE's raw channel H_j, nor that
    it knows H_k and P separately.  The receiver suppresses the other scheduled
    streams using the local interference-plus-noise covariance

        C_k = sum_{j != k} g_{k,j} g_{k,j}^H + sigma^2 I.

    The resulting SINR is the generalized Rayleigh quotient

        SINR_k = g_{k,k}^H C_k^{-1} g_{k,k}.

    This corresponds to a perfect-DMRS local LMMSE/SINR-maximizing combiner.
    """
    num_users = user_channels.shape[0]
    num_rx_antennas = user_channels.shape[1]
    sinr = np.zeros(num_users, dtype=np.float64)
    eye_rx = np.eye(num_rx_antennas, dtype=np.complex128)

    for k in range(num_users):
        hk = user_channels[k]

        # Local effective post-precoding channels available at UE k via DMRS:
        # G_k[:, j] = H_k p_j.
        g_eff = hk @ precoder
        desired_vec = g_eff[:, k]

        cov_int_noise = noise_power * eye_rx.copy()
        for j in range(num_users):
            if j != k:
                gj = g_eff[:, j]
                cov_int_noise += np.outer(gj, gj.conj())

        try:
            lmmse_direction = np.linalg.solve(cov_int_noise, desired_vec)
        except np.linalg.LinAlgError:
            lmmse_direction = np.linalg.pinv(cov_int_noise) @ desired_vec

        sinr[k] = max(float(np.real(np.vdot(desired_vec, lmmse_direction))), 0.0)

    return float(np.sum(np.log2(1.0 + np.maximum(sinr, eps)))), sinr


CQI_SINR_DB_LEVELS = np.array([0.2, 4.3, 5.9, 8.1, 10.3, 14.1, 18.7, 21.0], dtype=np.float64)


def quantize_ue_sinr_to_cqi_linear(sinr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Map linear per-UE SINR to the nearest CQI SINR report level."""
    sinr_db = 10.0 * np.log10(np.maximum(np.asarray(sinr, dtype=np.float64), eps))
    level_idx = np.argmin(np.abs(sinr_db[..., np.newaxis] - CQI_SINR_DB_LEVELS), axis=-1)
    return 10.0 ** (CQI_SINR_DB_LEVELS[level_idx] / 10.0)


def compute_cqi_quantized_sum_rate_from_sinr(sinr: np.ndarray, eps: float = 1e-12) -> float:
    quantized_sinr = quantize_ue_sinr_to_cqi_linear(sinr, eps=eps)
    return float(np.sum(np.log2(1.0 + np.maximum(quantized_sinr, eps))))



def compute_slot_sinr_proxy(
    user_channels: np.ndarray, precoder: np.ndarray, noise_power: float
) -> np.ndarray:
    """Compute a simple multi-user SINR proxy for reward shaping.

    This proxy uses the total received energy from each beam at each UE:

        desired_k = ||H_k p_k||_2^2
        interference_k = sum_{j != k} ||H_k p_j||_2^2
        proxy_sinr_k = desired_k / (interference_k + noise_power)

    It is different from compute_slot_sum_rate(), which now uses a local
    effective-channel LMMSE receiver based on G_k = H_k P. The proxy is
    intentionally simpler and gives the RL
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
    actual_sinr: np.ndarray,
    reference_actual_sinr: np.ndarray,
    eps: float,
) -> float:
    """Compute the scalar RL reward used in REINFORCE.

    Supported modes:
      - rate_log_ratio: continuous-valued sum-rate improvement over the reference.
      - cqi_reference_rate_log_ratio: continuous-valued sum-rate improvement over
        the reference after quantizing the reference UE SINRs to CQI levels.

    The default is rate_log_ratio so the policy is trained using a smooth
    objective that directly matches sum-rate improvement over the selected
    reference precoder.
    """
    if reward_mode == "rate_log_ratio":
        reference_sinr = np.asarray(reference_actual_sinr, dtype=np.float64)
    elif reward_mode == "cqi_reference_rate_log_ratio":
        reference_sinr = quantize_ue_sinr_to_cqi_linear(reference_actual_sinr, eps=eps)
    else:
        raise ValueError(
            f"Unknown reward_mode={reward_mode!r}. Expected one of: "
            "rate_log_ratio, cqi_reference_rate_log_ratio."
        )
    return float(np.sum(np.log((1.0 + actual_sinr + eps) / (1.0 + reference_sinr + eps))))


def compute_reward_reference_rate(
    reward_mode: str,
    continuous_reference_rate: float,
    reference_actual_sinr: np.ndarray,
    eps: float,
) -> float:
    """Return the scalar reference rate used by reward-side bonuses."""
    if reward_mode == "rate_log_ratio":
        return float(continuous_reference_rate)
    if reward_mode == "cqi_reference_rate_log_ratio":
        return compute_cqi_quantized_sum_rate_from_sinr(reference_actual_sinr, eps=eps)
    raise ValueError(
        f"Unknown reward_mode={reward_mode!r}. Expected one of: "
        "rate_log_ratio, cqi_reference_rate_log_ratio."
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



def simulate_toy_ar1_channels(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Original toy channel: independent coefficients with AR(1) time correlation.

    This is useful as a cheap fallback, but it has no explicit cross-antenna or
    cross-UE spatial structure.  Each scalar coefficient evolves independently
    with the same temporal correlation coefficient cfg.temporal_correlation.
    """
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


def _rough_rectangular_panel_dims(num_antennas: int) -> tuple[int, int]:
    """Choose a compact rectangular single-polarized panel with >= num_antennas elements."""
    n = max(1, int(num_antennas))
    rows = int(np.floor(np.sqrt(n)))
    rows = max(1, rows)
    cols = int(np.ceil(n / rows))
    return rows, cols


def _make_sionna_panel_array(num_antennas: int, carrier_frequency_hz: float, is_bs: bool):
    """Build a simple single-polarized Sionna PanelArray.

    Sionna's PanelArray wants a rectangular panel.  For antenna counts that are
    not exactly rectangular, we build the nearest larger panel and slice the
    generated channel down to the requested antenna count afterward.
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


def _sample_sionna_topology(cfg: SimConfig, rng: np.random.Generator):
    """Sample one BS and K UE positions/velocities for a Sionna topology."""
    num_users = cfg.num_users
    min_r = max(1.0, float(cfg.min_ue_distance_m))
    max_r = max(min_r + 1.0, float(cfg.cell_radius_m))
    # Uniform in area over an annulus.
    radii = np.sqrt(rng.uniform(min_r**2, max_r**2, size=num_users))
    angles = rng.uniform(-np.pi, np.pi, size=num_users)
    ut_loc = np.zeros((1, num_users, 3), dtype=np.float32)
    ut_loc[0, :, 0] = radii * np.cos(angles)
    ut_loc[0, :, 1] = radii * np.sin(angles)
    ut_loc[0, :, 2] = float(cfg.ue_height_m)

    bs_loc = np.array([[[0.0, 0.0, float(cfg.bs_height_m)]]], dtype=np.float32)
    ut_orientations = np.zeros((1, num_users, 3), dtype=np.float32)
    bs_orientations = np.zeros((1, 1, 3), dtype=np.float32)

    speed_mps = float(cfg.ue_speed_kmh) / 3.6
    velocity_angles = rng.uniform(-np.pi, np.pi, size=num_users)
    ut_velocities = np.zeros((1, num_users, 3), dtype=np.float32)
    ut_velocities[0, :, 0] = speed_mps * np.cos(velocity_angles)
    ut_velocities[0, :, 1] = speed_mps * np.sin(velocity_angles)

    # Keep all UEs outdoor. This avoids adding another random indoor/O2I state
    # unless the user explicitly wants to model it later.
    in_state = np.zeros((1, num_users), dtype=bool)
    return ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state


def simulate_sionna_channels(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate flat per-slot MIMO channels from Sionna TR 38.901 models.

    The returned tensor keeps this script's original narrowband interface:
        [num_slots, num_users, num_rx_antennas_per_user, num_tx_antennas].

    Internally, Sionna generates a time-evolving multipath channel impulse
    response for a fixed random topology. We convert the CIR to an OFDM channel
    and use the center subcarrier as the narrowband equivalent for this toy RL
    testbench.
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
            "Sionna channel generation requested, but Sionna/TensorFlow could not be imported. "
            "Install Sionna in this environment or rerun with --channel-model toy_ar1."
        ) from exc

    tf.random.set_seed(int(cfg.seed))
    ut_array = _make_sionna_panel_array(
        cfg.num_rx_antennas_per_user,
        cfg.carrier_frequency_hz,
        is_bs=False,
    )
    bs_array = _make_sionna_panel_array(
        cfg.num_tx_antennas,
        cfg.carrier_frequency_hz,
        is_bs=True,
    )

    model_name = cfg.sionna_scenario.lower()
    if cfg.channel_model.startswith("sionna_"):
        model_name = cfg.channel_model.split("sionna_", 1)[1].lower()
    model_cls = {"umi": UMi, "uma": UMa, "rma": RMa}.get(model_name)
    if model_cls is None:
        raise ValueError("Sionna channel model must be one of sionna_umi, sionna_uma, or sionna_rma.")

    channel_model = model_cls(
        carrier_frequency=float(cfg.carrier_frequency_hz),
        o2i_model=str(cfg.sionna_o2i_model),
        ut_array=ut_array,
        bs_array=bs_array,
        direction="downlink",
        enable_pathloss=bool(cfg.sionna_enable_pathloss),
        enable_shadow_fading=bool(cfg.sionna_enable_shadow_fading),
    )

    topo = _sample_sionna_topology(cfg, rng)
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
        f"Generating Sionna {model_name.upper()} channels: "
        f"K={cfg.num_users}, Nt={cfg.num_tx_antennas}, "
        f"Nrx={cfg.num_rx_antennas_per_user}, slots={cfg.num_slots}, "
        f"fc={cfg.carrier_frequency_hz/1e9:.3f} GHz, speed={cfg.ue_speed_kmh:.2f} km/h"
    )
    a, tau = channel_model(
        num_time_samples=int(cfg.num_slots),
        sampling_frequency=float(sampling_frequency),
    )

    # Use a tiny 3-subcarrier grid and select the center RE. This avoids any
    # ambiguity about whether a one-subcarrier frequency grid is exactly centered
    # at 0 Hz across Sionna versions.
    freqs = subcarrier_frequencies(3, float(cfg.subcarrier_spacing_hz))
    h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=True)
    h_np = np.asarray(h_freq.numpy())

    # Expected Sionna shape:
    # [batch, num_rx/UT, rx_ant, num_tx/BS, tx_ant, num_time_samples, num_subcarriers].
    if h_np.ndim != 7:
        raise RuntimeError(f"Unexpected Sionna OFDM channel rank {h_np.ndim}; shape={h_np.shape}")
    h_center = h_np[0, : cfg.num_users, : cfg.num_rx_antennas_per_user, 0, : cfg.num_tx_antennas, :, 1]
    channels = np.transpose(h_center, (3, 0, 1, 2)).astype(np.complex128, copy=False)
    expected_shape = (
        cfg.num_slots,
        cfg.num_users,
        cfg.num_rx_antennas_per_user,
        cfg.num_tx_antennas,
    )
    if channels.shape != expected_shape:
        raise RuntimeError(f"Unexpected generated channel shape {channels.shape}; expected {expected_shape}.")
    return channels


def simulate_channels(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.channel_model == "toy_ar1":
        return simulate_toy_ar1_channels(cfg, rng)
    if cfg.channel_model in {"sionna_umi", "sionna_uma", "sionna_rma"}:
        return simulate_sionna_channels(cfg, rng)
    raise ValueError("--channel-model must be one of: toy_ar1, sionna_umi, sionna_uma, sionna_rma.")


def run_zf_baseline(
    cfg: SimConfig, channels: np.ndarray, pmi_feedback: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    sinr_trace = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.float64)
    precoders = np.zeros(
        (cfg.num_slots, cfg.num_tx_antennas, cfg.num_users), dtype=np.complex128
    )
    if pmi_feedback is None:
        pmi_feedback = build_pmi_feedback_trace(cfg, channels)

    for t in range(cfg.num_slots):
        print(f"ZF Slot {t + 1} / {cfg.num_slots}", end="\r")
        p_zf = build_zf_precoder_from_q_vectors(
            pmi_feedback["q_vectors"][t], cfg.total_tx_power
        )
        throughput[t], sinr_trace[t] = compute_slot_sum_rate(channels[t], p_zf, noise_power)
        precoders[t] = p_zf

    print()
    return {
        "throughput": throughput,
        "sinr": sinr_trace,
        "precoders": precoders,
        "pmi_features": pmi_feedback["features"],
    }



def run_slnr_baseline(
    cfg: SimConfig, channels: np.ndarray, pmi_feedback: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
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

    if pmi_feedback is None:
        pmi_feedback = build_pmi_feedback_trace(cfg, channels)

    for t in range(cfg.num_slots):
        print(f"SLNR Slot {t + 1} / {cfg.num_slots}", end="\r")
        p_slnr = build_slnr_precoder_from_q_vectors(
            pmi_feedback["q_vectors"][t], cfg.total_tx_power, noise_power
        )
        throughput[t], sinr_trace[t] = compute_slot_sum_rate(channels[t], p_slnr, noise_power)
        precoders[t] = p_slnr

    print()
    return {
        "throughput": throughput,
        "sinr": sinr_trace,
        "precoders": precoders,
    }


def compute_slot_sum_rate_mmse_receiver(
    user_channels: np.ndarray,
    precoder: np.ndarray,
    noise_power: float,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Compute the single-stream-per-UE sum-rate with the SINR-maximizing linear receiver.

    For UE k, the receiver is allowed to use the perfect local effective channel
    G_k = H_k P, equivalently computed here from H_k and P for simulation.
    The resulting SINR is

        SINR_k = p_k^H H_k^H C_k^{-1} H_k p_k,

    where C_k = sum_{j != k} H_k p_j p_j^H H_k^H + sigma^2 I is the
    interference-plus-noise covariance.  This is the rate expression matched to
    the WMMSE updates below. It matches the local perfect-DMRS LMMSE receiver
    now used by compute_slot_sum_rate().
    """
    num_users = user_channels.shape[0]
    num_rx_antennas = user_channels.shape[1]
    sinr = np.zeros(num_users, dtype=np.float64)

    eye_rx = np.eye(num_rx_antennas, dtype=np.complex128)
    for k in range(num_users):
        hk = user_channels[k]
        pk = precoder[:, k]
        desired_vec = hk @ pk
        cov_int_noise = noise_power * eye_rx.copy()
        for j in range(num_users):
            if j == k:
                continue
            hjpj = hk @ precoder[:, j]
            cov_int_noise += np.outer(hjpj, hjpj.conj())
        try:
            solved = np.linalg.solve(cov_int_noise, desired_vec)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(cov_int_noise) @ desired_vec
        sinr[k] = max(float(np.real(np.vdot(desired_vec, solved))), 0.0)

    return float(np.sum(np.log2(1.0 + np.maximum(sinr, eps)))), sinr


def build_full_csi_mrt_initial_precoder(
    user_channels: np.ndarray,
    total_tx_power: float,
) -> np.ndarray:
    """Initialize WMMSE with full-CSI dominant right singular vectors.

    This uses the true H_k, not PMI feedback, and is only an initialization for
    the non-convex WMMSE iterations.
    """
    num_users, _, num_tx_antennas = user_channels.shape
    directions = np.zeros((num_tx_antennas, num_users), dtype=np.complex128)
    for k in range(num_users):
        _, _, vh = np.linalg.svd(user_channels[k], full_matrices=True)
        directions[:, k] = vh.conj().T[:, 0]
    return normalize_columns_equal_power(directions, total_tx_power)


def _solve_wmmse_precoder_update(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    total_tx_power: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Solve the WMMSE transmit update with a sum-power constraint.

    The update has the closed form P(lambda) = (A + lambda I)^{-1} B.  The
    scalar lambda >= 0 is chosen so that ||P(lambda)||_F^2 <= total_tx_power,
    with equality unless the unconstrained lambda=0 solution already satisfies
    the power constraint.
    """
    num_tx_antennas = a_mat.shape[0]
    eye_tx = np.eye(num_tx_antennas, dtype=np.complex128)

    def solve_for_lambda(lam: float) -> tuple[np.ndarray, float]:
        mat = a_mat + float(lam) * eye_tx
        try:
            p_lam = np.linalg.solve(mat, b_mat)
        except np.linalg.LinAlgError:
            p_lam = np.linalg.pinv(mat) @ b_mat
        power = float(np.real(np.sum(np.abs(p_lam) ** 2)))
        return p_lam, power

    p0, power0 = solve_for_lambda(0.0)
    if power0 <= total_tx_power or power0 <= eps:
        return p0

    lam_low = 0.0
    lam_high = 1.0
    _, power_high = solve_for_lambda(lam_high)
    while power_high > total_tx_power:
        lam_high *= 2.0
        _, power_high = solve_for_lambda(lam_high)
        if lam_high > 1e12:
            break

    p_mid = p0
    for _ in range(60):
        lam_mid = 0.5 * (lam_low + lam_high)
        p_mid, power_mid = solve_for_lambda(lam_mid)
        if power_mid > total_tx_power:
            lam_low = lam_mid
        else:
            lam_high = lam_mid
    return p_mid


def build_wmmse_precoder_full_csi(
    user_channels: np.ndarray,
    total_tx_power: float,
    noise_power: float,
    max_iters: int = 50,
    tol: float = 1e-5,
    eps: float = 1e-12,
) -> np.ndarray:
    """Weighted-MMSE precoder using perfect full channel matrices H_k.

    This implements the standard single-stream-per-user WMMSE block coordinate
    descent for the weighted sum-rate objective with equal user weights:

        maximize_P  sum_k log2(1 + SINR_k(P))
        subject to  sum_k ||p_k||_2^2 <= total_tx_power.

    It is a full-CSI oracle baseline and does not use PMI, Type-II feedback, or
    any WESN/RL state.
    """
    num_users, num_rx_antennas, num_tx_antennas = user_channels.shape
    p_mat = build_full_csi_mrt_initial_precoder(user_channels, total_tx_power)
    eye_rx = np.eye(num_rx_antennas, dtype=np.complex128)

    prev_rate = -np.inf
    for _ in range(max(1, int(max_iters))):
        # Receiver and MSE-weight update.
        u_list: list[np.ndarray] = []
        w_vec = np.zeros(num_users, dtype=np.float64)
        for k in range(num_users):
            hk = user_channels[k]
            y_cov = noise_power * eye_rx.copy()
            for j in range(num_users):
                hjpj = hk @ p_mat[:, j]
                y_cov += np.outer(hjpj, hjpj.conj())
            desired_vec = hk @ p_mat[:, k]
            try:
                u_k = np.linalg.solve(y_cov, desired_vec)
            except np.linalg.LinAlgError:
                u_k = np.linalg.pinv(y_cov) @ desired_vec
            mse_k = 1.0 - 2.0 * np.real(np.vdot(u_k, desired_vec)) + np.real(np.vdot(u_k, y_cov @ u_k))
            mse_k = max(float(mse_k), eps)
            u_list.append(u_k)
            w_vec[k] = 1.0 / mse_k

        # Transmit precoder update.
        a_mat = np.zeros((num_tx_antennas, num_tx_antennas), dtype=np.complex128)
        b_mat = np.zeros((num_tx_antennas, num_users), dtype=np.complex128)
        for k in range(num_users):
            hk = user_channels[k]
            u_k = u_list[k]
            hk_h_u = hk.conj().T @ u_k
            a_mat += w_vec[k] * np.outer(hk_h_u, hk_h_u.conj())
            b_mat[:, k] = w_vec[k] * hk_h_u
        p_new = _solve_wmmse_precoder_update(a_mat, b_mat, total_tx_power, eps=eps)

        rate, _ = compute_slot_sum_rate_mmse_receiver(user_channels, p_new, noise_power, eps=eps)
        p_mat = p_new
        if np.isfinite(prev_rate) and abs(rate - prev_rate) <= float(tol) * max(1.0, abs(prev_rate)):
            break
        prev_rate = rate

    return p_mat


def run_wmmse_baseline(
    cfg: SimConfig,
    channels: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-5,
) -> dict[str, np.ndarray]:
    """Run the perfect-full-CSI WMMSE performance-ceiling baseline."""
    if cfg.streams_per_user != 1:
        raise ValueError("This WMMSE baseline currently assumes one stream per UE.")

    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    sinr_trace = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.float64)
    precoders = np.zeros(
        (cfg.num_slots, cfg.num_tx_antennas, cfg.num_users), dtype=np.complex128
    )

    for t in range(cfg.num_slots):
        print(f"WMMSE full-CSI Slot {t + 1} / {cfg.num_slots}", end="\r")
        p_wmmse = build_wmmse_precoder_full_csi(
            channels[t],
            total_tx_power=cfg.total_tx_power,
            noise_power=noise_power,
            max_iters=max_iters,
            tol=tol,
        )
        throughput[t], sinr_trace[t] = compute_slot_sum_rate_mmse_receiver(
            channels[t], p_wmmse, noise_power
        )
        precoders[t] = p_wmmse

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


def angle_to_raw_sigmoid_coordinate(value: float, max_value: float, eps: float = 1e-6) -> float:
    """Convert a bounded angle value to the corresponding raw sigmoid coordinate.

    If angle = max_value * sigmoid(raw), then this returns raw.
    """
    ratio = float(value) / max(float(max_value), eps)
    ratio = float(np.clip(ratio, eps, 1.0 - eps))
    return float(np.log(ratio / (1.0 - ratio)))


def build_initial_gaussian_raw_bias(rl_cfg: RLConfig, num_users: int) -> np.ndarray:
    """Deprecated constant raw bias initializer retained for compatibility.

    The active Gaussian PMI-span policy now uses a slot-dependent, PMI-based
    leakage-nulling center, not this constant oracle-like bias.
    """
    raw_theta = angle_to_raw_sigmoid_coordinate(
        value=rl_cfg.gaussian_init_theta,
        max_value=rl_cfg.pmi_span_theta_max,
    )
    raw_phi = angle_to_raw_sigmoid_coordinate(
        value=rl_cfg.gaussian_init_phi,
        max_value=2.0 * np.pi,
    )
    return np.tile(np.array([raw_theta, raw_phi], dtype=np.float64), int(num_users))


def wrap_angle_0_2pi(x: float) -> float:
    return float(np.mod(float(x), 2.0 * np.pi))


def pmi_leakage_nulling_theta_phi_center(
    q_vectors: np.ndarray,
    theta_max: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return causal PMI-span center angles that null PMI overlap for K=2.

    For UE k, with q_o denoting the other user's PMI vector, write

        q_o = c q_k + beta b_k,

    where b_k is the normalized component of q_o orthogonal to q_k and
    c = q_k^H q_o.  The PMI-span beam is

        v_k(theta, phi) = cos(theta) q_k + exp(j phi) sin(theta) b_k.

    Choosing

        tan(theta) = |c| / beta,
        phi = angle(-q_o^H q_k),

    makes q_o^H v_k approximately zero.  This gives a state-dependent,
    leakage-aware center using only current PMI vectors, so SLNR remains only
    an external baseline and is not inserted as a learned action.
    """
    if q_vectors.shape[0] != 2:
        raise ValueError("The PMI leakage-nulling center currently implements K=2 only.")

    centers = np.zeros((2, 2), dtype=np.float64)
    for k in range(2):
        qk = unit_norm(q_vectors[k])
        qo = unit_norm(q_vectors[1 - k])

        # q_o^H q_k is the coefficient appearing directly in q_o^H v_k.
        c_other_h_own = np.vdot(qo, qk)
        abs_c = float(np.abs(c_other_h_own))
        beta = float(np.sqrt(max(1.0 - abs_c**2, 0.0)))

        if beta < eps or abs_c < eps:
            theta = 0.0
            phi = 0.0
        else:
            theta = float(np.arctan2(abs_c, beta))
            theta = float(np.clip(theta, 0.0, float(theta_max)))
            phi = wrap_angle_0_2pi(np.angle(-c_other_h_own))

        centers[k, 0] = theta
        centers[k, 1] = phi
    return centers


def pmi_slnr_theta_phi_center(
    q_vectors: np.ndarray,
    slnr_precoder: np.ndarray,
    theta_max: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return PMI-span coordinates of the PMI-SLNR precoder for K=2.

    For each UE k, the Gaussian policy uses the user-centered PMI-span beam

        v_k(theta, phi) = cos(theta) q_k + exp(j phi) sin(theta) b_k,

    where b_k is the component of the other user's PMI direction orthogonal to
    q_k.  This function projects the already-computed PMI-SLNR beam onto this
    same {q_k, b_k} coordinate system and returns its physical [theta, phi]
    angles.  The resulting center is causal because the SLNR beam is built from
    the current PMI vectors only; it is not inserted as a selectable candidate.
    """
    if q_vectors.shape[0] != 2:
        raise ValueError("The PMI-SLNR angle center currently implements K=2 only.")

    centers = np.zeros((2, 2), dtype=np.float64)
    for k in range(2):
        qk = unit_norm(q_vectors[k])
        qo = unit_norm(q_vectors[1 - k])

        b_tilde = qo - qk * np.vdot(qk, qo)
        b_norm = float(np.linalg.norm(b_tilde))
        if b_norm < eps:
            centers[k, 0] = 0.0
            centers[k, 1] = 0.0
            continue
        bk = b_tilde / b_norm

        # Remove the equal-power amplitude and arbitrary global phase.  The
        # global phase is chosen so q_k^H v is real/nonnegative, matching the
        # theta/phi parameterization used by user_centered_pmi_span_beam_from_theta_phi().
        v = unit_norm(slnr_precoder[:, k])
        c0 = np.vdot(qk, v)
        if np.abs(c0) > eps:
            v = np.exp(-1j * np.angle(c0)) * v

        a0 = np.vdot(qk, v)
        a1 = np.vdot(bk, v)
        theta = float(np.arctan2(np.abs(a1), np.abs(a0)))
        theta = float(np.clip(theta, 0.0, float(theta_max)))
        phi = wrap_angle_0_2pi(np.angle(a1)) if np.abs(a1) > eps else 0.0

        centers[k, 0] = theta
        centers[k, 1] = phi
    return centers



def pmi_proxy_sum_rate_from_theta_phi(
    q_vectors: np.ndarray,
    theta_phi: np.ndarray,
    total_tx_power: float,
    noise_power: float,
    eps: float = 1e-12,
) -> float:
    """PMI-only proxy sum-rate for a PMI-span angle vector.

    This uses only the dominant PMI vectors q_k and the same equal-power
    normalization as the executed precoder.  It is used only to compute a causal
    direction in angle space; the RL reward is still computed from the actual
    channel and the executed precoder.
    """
    num_users = q_vectors.shape[0]
    num_tx_antennas = q_vectors.shape[1]
    tp = np.asarray(theta_phi, dtype=np.float64).reshape(num_users, 2)
    beams = np.zeros((num_tx_antennas, num_users), dtype=np.complex128)
    for k in range(num_users):
        beam_k = user_centered_pmi_span_beam_from_theta_phi(
            q_vectors=q_vectors,
            user_index=k,
            theta=float(tp[k, 0]),
            phi=float(tp[k, 1]),
        )
        beams[:, k] = np.sqrt(total_tx_power / max(num_users, 1)) * beam_k
    proxy_sinr = compute_pmi_sinr_proxy_from_precoder(q_vectors, beams, noise_power, eps=eps)
    return float(np.sum(np.log1p(proxy_sinr)))



def pmi_proxy_signal_leakage_from_theta_phi(
    q_vectors: np.ndarray,
    theta_phi: np.ndarray,
    total_tx_power: float,
    noise_power: float,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """Return PMI-only (proxy-rate, desired-signal, leakage) for angle coordinates.

    The signal term is sum_k |q_k^H p_k|^2. The leakage term is
    sum_k sum_{j != k} |q_j^H p_k|^2, i.e., leakage created by each transmit
    beam into the other PMI directions. These are used only to construct
    causal search directions in the PMI-span angle coordinates; the executed
    RL reward still uses the true channel sum-rate.
    """
    num_users = q_vectors.shape[0]
    num_tx_antennas = q_vectors.shape[1]
    tp = np.asarray(theta_phi, dtype=np.float64).reshape(num_users, 2)
    beams = np.zeros((num_tx_antennas, num_users), dtype=np.complex128)
    for k in range(num_users):
        beam_k = user_centered_pmi_span_beam_from_theta_phi(
            q_vectors=q_vectors,
            user_index=k,
            theta=float(tp[k, 0]),
            phi=float(tp[k, 1]),
        )
        beams[:, k] = np.sqrt(total_tx_power / max(num_users, 1)) * beam_k

    signal_sum = 0.0
    leakage_sum = 0.0
    for tx_k in range(num_users):
        signal_sum += float(np.abs(np.vdot(q_vectors[tx_k], beams[:, tx_k])) ** 2)
        for rx_j in range(num_users):
            if rx_j != tx_k:
                leakage_sum += float(np.abs(np.vdot(q_vectors[rx_j], beams[:, tx_k])) ** 2)

    proxy_sinr = compute_pmi_sinr_proxy_from_precoder(q_vectors, beams, noise_power, eps=eps)
    proxy_rate = float(np.sum(np.log1p(proxy_sinr)))
    return proxy_rate, float(signal_sum), float(leakage_sum)


def finite_difference_angle_gradient_direction(
    objective_fn,
    center_theta_phi: np.ndarray,
    theta_max: float,
    eps: float = 1e-12,
    sign: float = 1.0,
    fd_step_theta: float = 1e-4,
    fd_step_phi: float = 1e-4,
) -> np.ndarray:
    """Finite-difference normalized direction in [theta0, phi0, theta1, phi1]."""
    num_users = center_theta_phi.shape[0]
    flat = np.asarray(center_theta_phi, dtype=np.float64).reshape(-1)
    grad = np.zeros_like(flat)
    for i in range(flat.size):
        delta = fd_step_theta if (i % 2 == 0) else fd_step_phi
        step = np.zeros_like(flat)
        step[i] = delta
        plus = apply_angle_step_from_flat(flat, step, num_users, theta_max)
        minus = apply_angle_step_from_flat(flat, -step, num_users, theta_max)
        grad[i] = (float(objective_fn(plus)) - float(objective_fn(minus))) / max(2.0 * delta, eps)
    grad = float(sign) * grad
    nrm = float(np.linalg.norm(grad))
    if nrm < eps or not np.all(np.isfinite(grad)):
        return np.zeros_like(flat)
    return grad / nrm


def pmi_proxy_direction_dictionary_theta_phi(
    q_vectors: np.ndarray,
    center_theta_phi: np.ndarray,
    total_tx_power: float,
    noise_power: float,
    theta_max: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, list[str]]:
    """Build the angle-space direction dictionary used by the multi-direction policy.

    Directions are:
      0) proxy-rate ascent:        +grad R_PMI
      1) leakage-reduction:        -grad L_PMI
      2) desired-signal increase:  +grad S_PMI
    All directions are normalized in the 4D physical angle coordinate vector.
    """
    def rate_obj(tp: np.ndarray) -> float:
        r, _, _ = pmi_proxy_signal_leakage_from_theta_phi(
            q_vectors=q_vectors,
            theta_phi=tp,
            total_tx_power=total_tx_power,
            noise_power=noise_power,
            eps=eps,
        )
        return r

    def signal_obj(tp: np.ndarray) -> float:
        _, s, _ = pmi_proxy_signal_leakage_from_theta_phi(
            q_vectors=q_vectors,
            theta_phi=tp,
            total_tx_power=total_tx_power,
            noise_power=noise_power,
            eps=eps,
        )
        return s

    def leakage_obj(tp: np.ndarray) -> float:
        _, _, ell = pmi_proxy_signal_leakage_from_theta_phi(
            q_vectors=q_vectors,
            theta_phi=tp,
            total_tx_power=total_tx_power,
            noise_power=noise_power,
            eps=eps,
        )
        return ell

    d_rate = finite_difference_angle_gradient_direction(rate_obj, center_theta_phi, theta_max, eps=eps, sign=1.0)
    d_leak = finite_difference_angle_gradient_direction(leakage_obj, center_theta_phi, theta_max, eps=eps, sign=-1.0)
    d_sig = finite_difference_angle_gradient_direction(signal_obj, center_theta_phi, theta_max, eps=eps, sign=1.0)
    return np.stack([d_rate, d_leak, d_sig], axis=0), ["proxy-rate", "leakage-reducing", "signal-increasing"]

def apply_angle_step_from_flat(
    center_flat: np.ndarray,
    step_flat: np.ndarray,
    num_users: int,
    theta_max: float,
) -> np.ndarray:
    """Apply a flat 4D angle step, clipping theta and wrapping phi."""
    x = np.asarray(center_flat, dtype=np.float64).reshape(num_users, 2).copy()
    step = np.asarray(step_flat, dtype=np.float64).reshape(num_users, 2)
    x[:, 0] = np.clip(x[:, 0] + step[:, 0], 0.0, float(theta_max))
    x[:, 1] = np.mod(x[:, 1] + step[:, 1], 2.0 * np.pi)
    return x


def pmi_proxy_rate_gradient_direction_theta_phi(
    q_vectors: np.ndarray,
    center_theta_phi: np.ndarray,
    total_tx_power: float,
    noise_power: float,
    theta_max: float,
    eps: float = 1e-12,
    fd_step_theta: float = 1e-4,
    fd_step_phi: float = 1e-4,
) -> np.ndarray:
    """Return normalized 4D gradient direction of a PMI-only proxy rate.

    The gradient is with respect to the physical angle vector
    [theta0, phi0, theta1, phi1].  It is evaluated at the SLNR angle center.
    This provides a causal, geometry-aware direction; the WESN learns only the
    diagonal step sizes along this direction.
    """
    num_users = q_vectors.shape[0]
    center = np.asarray(center_theta_phi, dtype=np.float64).reshape(num_users, 2)
    flat = center.reshape(-1)
    grad = np.zeros_like(flat)
    for i in range(flat.size):
        delta = fd_step_theta if (i % 2 == 0) else fd_step_phi
        step = np.zeros_like(flat)
        step[i] = delta
        plus = apply_angle_step_from_flat(flat, step, num_users, theta_max)
        minus = apply_angle_step_from_flat(flat, -step, num_users, theta_max)
        r_plus = pmi_proxy_sum_rate_from_theta_phi(
            q_vectors=q_vectors,
            theta_phi=plus,
            total_tx_power=total_tx_power,
            noise_power=noise_power,
            eps=eps,
        )
        r_minus = pmi_proxy_sum_rate_from_theta_phi(
            q_vectors=q_vectors,
            theta_phi=minus,
            total_tx_power=total_tx_power,
            noise_power=noise_power,
            eps=eps,
        )
        grad[i] = (r_plus - r_minus) / max(2.0 * delta, eps)
    nrm = float(np.linalg.norm(grad))
    if nrm < eps or not np.all(np.isfinite(grad)):
        return np.zeros_like(flat)
    return grad / nrm

def theta_phi_to_raw_gaussian_action(
    theta_phi: np.ndarray,
    num_users: int,
    theta_max: float,
) -> np.ndarray:
    """Convert physical PMI-span angles [theta, phi] to raw sigmoid coordinates."""
    tp = np.asarray(theta_phi, dtype=np.float64).reshape(num_users, 2)
    raw = np.zeros((num_users, 2), dtype=np.float64)
    for k in range(num_users):
        raw[k, 0] = angle_to_raw_sigmoid_coordinate(tp[k, 0], theta_max)
        raw[k, 1] = angle_to_raw_sigmoid_coordinate(wrap_angle_0_2pi(tp[k, 1]), 2.0 * np.pi)
    return raw.reshape(-1)


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


def phase_align_precoder_columns_to_reference(
    reference_precoder: np.ndarray,
    learned_precoder: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Phase-align each learned precoder column to the matching reference column.

    The multi-user SINR is invariant to an independent complex phase rotation
    on each stream/beam.  Before measuring whole-precoder distance, remove this
    harmless per-column phase difference so the distance reflects beam/precoder
    geometry rather than arbitrary stream phase.
    """
    aligned = np.asarray(learned_precoder, dtype=np.complex128).copy()
    reference = np.asarray(reference_precoder, dtype=np.complex128)
    if aligned.shape != reference.shape:
        raise ValueError(
            f"reference_precoder and learned_precoder must have the same shape; "
            f"got {reference.shape} and {aligned.shape}."
        )
    for k in range(reference.shape[1]):
        inner = np.vdot(reference[:, k], aligned[:, k])
        if np.abs(inner) > eps:
            aligned[:, k] *= np.exp(-1j * np.angle(inner))
    return aligned


def phase_aligned_precoder_distance(
    reference_precoder: np.ndarray,
    learned_precoder: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Return normalized whole-precoder distance after per-column phase alignment."""
    aligned = phase_align_precoder_columns_to_reference(
        reference_precoder=reference_precoder,
        learned_precoder=learned_precoder,
        eps=eps,
    )
    denom = max(float(np.linalg.norm(reference_precoder, ord="fro")), eps)
    return float(np.linalg.norm(reference_precoder - aligned, ord="fro") / denom)


def reference_like_baseline_update_weight(
    precoder_distance: float,
    alpha_max: float,
    sigma_d: float,
    eps: float = 1e-12,
) -> float:
    """Convert reference/learned precoder distance into an EMA update weight.

    Near-reference executed precoders should update the reference-like baseline
    strongly; far perturbations should barely update it.
    """
    sigma = max(float(sigma_d), eps)
    d = max(float(precoder_distance), 0.0)
    return float(max(float(alpha_max), 0.0) * np.exp(-0.5 * (d / sigma) ** 2))


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


def parse_alpha_level_grid(s: str) -> tuple[float, ...]:
    """Parse comma-separated alpha-level multipliers for the angle-step policy.

    The levels are dimensionless multipliers applied to alpha_step_max_theta/phi.
    We force 0.0 to be present so the exact reference-precoder-center action remains available.
    """
    vals: list[float] = []
    for item in s.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        vals = [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]
    vals = sorted(set(vals))
    if 0.0 not in vals:
        vals.append(0.0)
        vals = sorted(set(vals))
    return tuple(vals)


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


def pmi_span_num_candidates(rl_cfg: RLConfig) -> int:
    """Number of user-centered PMI-span candidates per UE for K=2.

    Candidate 0 is an exact PMI-SLNR anchor. Candidate 1 is own PMI
    (theta=0), and the rest are the theta/phi grid points.
    """
    ntheta = max(1, int(rl_cfg.pmi_span_num_theta))
    nphi = max(1, int(rl_cfg.pmi_span_num_phi))
    return 2 + max(0, ntheta - 1) * nphi


def build_user_centered_pmi_span_candidate_pool_real(
    q_vectors: np.ndarray,
    user_index: int,
    rl_cfg: RLConfig,
    slnr_anchor: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed-size real-packed PMI-span candidate pool for one UE.

    For K=2, candidate beams are
        v_k(theta, phi) = cos(theta) q_k + exp(j phi) sin(theta) b_k,
    where b_k is q_other with the component parallel to q_k removed.

    Candidate 0 is the exact PMI-SLNR anchor when provided. This makes the
    learned branch's candidate set contain the baseline exactly, rather than
    relying on the theta/phi grid to hit it.

    Returns candidate_pool [M, 2*N_t] and theta_phi [M, 2].
    """
    if q_vectors.shape[0] != 2:
        raise ValueError("The user-centered PMI-span RL policy currently implements K=2 only.")
    qk = unit_norm(q_vectors[user_index])
    q_other = unit_norm(q_vectors[1 - user_index])
    ntheta = max(1, int(rl_cfg.pmi_span_num_theta))
    nphi = max(1, int(rl_cfg.pmi_span_num_phi))
    theta_grid = np.linspace(0.0, float(rl_cfg.pmi_span_theta_max), ntheta, endpoint=True)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)

    b_tilde = q_other - qk * np.vdot(qk, q_other)
    b_norm = float(np.linalg.norm(b_tilde))
    bk = np.zeros_like(qk) if b_norm < eps else b_tilde / b_norm

    def coords_in_user_basis(v_in: np.ndarray) -> tuple[float, float]:
        if b_norm < eps:
            return 0.0, 0.0
        v = unit_norm(v_in)
        a0 = np.vdot(qk, v)
        if np.abs(a0) > eps:
            v = np.exp(-1j * np.angle(a0)) * v
        c0 = np.vdot(qk, v)
        c1 = np.vdot(bk, v)
        theta = float(np.arctan2(np.abs(c1), np.abs(c0)))
        phi = float(np.mod(np.angle(c1), 2.0 * np.pi)) if np.abs(c1) > eps else 0.0
        return theta, phi

    cand: list[np.ndarray] = []
    params: list[tuple[float, float]] = []
    if slnr_anchor is None:
        cand.append(qk)
        params.append((0.0, 0.0))
    else:
        cand.append(unit_norm(slnr_anchor))
        params.append(coords_in_user_basis(slnr_anchor))

    # Exact own-PMI candidate, independent of phi.
    cand.append(qk)
    params.append((0.0, 0.0))
    for theta in theta_grid[1:]:
        for phi in phi_grid:
            v = qk if b_norm < eps else np.cos(theta) * qk + np.exp(1j * phi) * np.sin(theta) * bk
            cand.append(unit_norm(v))
            params.append((float(theta), float(phi)))
    return np.stack([complex_unit_to_real(c) for c in cand], axis=0), np.array(params, dtype=np.float64)


def sample_user_centered_pmi_span_policy(
    candidate_logits: np.ndarray,
    signal_mat: np.ndarray,
    leakage_mat: np.ndarray,
    signal_gamma: float,
    slnr_noise_power: float,
    rng: np.random.Generator,
    leakage_norm_eps: float,
    q_vectors: np.ndarray,
    user_index: int,
    candidate_temperature: float,
    rl_cfg: RLConfig,
    slnr_anchor_complex: np.ndarray | None = None,
) -> tuple[
    np.ndarray, float, float, int, bool, np.ndarray, np.ndarray, np.ndarray, int, dict[str, float], np.ndarray, np.ndarray
]:
    """Sample one beam from the user-centered PMI-span categorical policy.

    WESN outputs one logit per theta/phi candidate. Conditioned on the current
    PMI vectors and deterministic candidate pool, REINFORCE uses the categorical
    log-probability of the selected index.
    """
    candidate_pool, theta_phi = build_user_centered_pmi_span_candidate_pool_real(
        q_vectors=q_vectors,
        user_index=user_index,
        rl_cfg=rl_cfg,
        slnr_anchor=slnr_anchor_complex,
    )
    num_candidates = candidate_pool.shape[0]
    logits_raw = np.asarray(candidate_logits, dtype=np.float64).reshape(-1)
    if logits_raw.size != num_candidates:
        raise ValueError(f"candidate_logits has size {logits_raw.size}, expected {num_candidates}")

    noise_floor = max(float(slnr_noise_power), leakage_norm_eps)
    signal_pool = np.zeros(num_candidates, dtype=np.float64)
    leakage_pool = np.zeros(num_candidates, dtype=np.float64)
    slnr_score_pool = np.zeros(num_candidates, dtype=np.float64)
    for i in range(num_candidates):
        x = candidate_pool[i]
        g = raw_quadratic_score(x, signal_mat)
        ell = raw_quadratic_score(x, leakage_mat)
        signal_pool[i] = g
        leakage_pool[i] = ell
        slnr_score_pool[i] = signal_gamma * g / max(ell + noise_floor, leakage_norm_eps)

    logits = logits_raw / max(float(candidate_temperature), leakage_norm_eps)
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
        "selected_policy_score": float(logits_raw[selected_idx]),
        "best_policy_score": float(np.max(logits_raw)),
        "mean_policy_score": float(np.mean(logits_raw)),
        "selected_minus_mean_policy_score": float(logits_raw[selected_idx] - np.mean(logits_raw)),
        "phase_score_mean": float(np.mean(logits_raw)),
        "phase_score_std": float(np.std(logits_raw)),
        "slnr_score_mean": float(np.mean(slnr_score_pool)),
        "slnr_score_std": float(np.std(slnr_score_pool)),
        "policy_score_std": float(np.std(logits_raw)),
        "selected_source": 2.0 if selected_idx == 0 else 4.0,
    }
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
        candidate_pool[0],
        theta_phi[selected_idx],
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


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def raw_gaussian_action_to_theta_phi(
    raw_action: np.ndarray, num_users: int, theta_max: float
) -> np.ndarray:
    """Map unconstrained raw Gaussian action to bounded [theta, phi] angles.

    raw_action has shape [2*K] and is interpreted as
    [u_theta0, u_phi0, u_theta1, u_phi1, ...].  The returned array has
    shape [K, 2], with theta in [0, theta_max] and phi in [0, 2*pi].
    """
    raw = np.asarray(raw_action, dtype=np.float64).reshape(num_users, 2)
    sig = sigmoid_np(raw)
    theta = float(theta_max) * sig[:, 0]
    phi = 2.0 * np.pi * sig[:, 1]
    return np.stack([theta, phi], axis=1)


def user_centered_pmi_span_beam_from_theta_phi(
    q_vectors: np.ndarray,
    user_index: int,
    theta: float,
    phi: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build v_k(theta,phi) in the K=2 user-centered PMI span.

    v_k(theta, phi) = cos(theta) q_k + exp(j phi) sin(theta) b_k,
    where b_k is the other UE's PMI vector after removing the component
    parallel to q_k.
    """
    if q_vectors.shape[0] != 2:
        raise ValueError("The 4D Gaussian PMI-span policy currently implements K=2 only.")
    qk = unit_norm(q_vectors[user_index])
    q_other = unit_norm(q_vectors[1 - user_index])
    b_tilde = q_other - qk * np.vdot(qk, q_other)
    b_norm = float(np.linalg.norm(b_tilde))
    if b_norm < eps:
        return qk
    bk = b_tilde / b_norm
    v = np.cos(float(theta)) * qk + np.exp(1j * float(phi)) * np.sin(float(theta)) * bk
    return unit_norm(v)


def gaussian_raw_log_prob_torch(
    raw_action: torch.Tensor,
    raw_mean: torch.Tensor,
    std_per_dim: torch.Tensor,
) -> torch.Tensor:
    """Log probability of a diagonal Gaussian raw action.

    raw_action and raw_mean have shape [B, 2K]. std_per_dim has shape [2K].
    Returns log pi(a|s) with shape [B].
    """
    std = torch.clamp(std_per_dim, min=1e-12).unsqueeze(0)
    z = (raw_action - raw_mean) / std
    log_probs = -0.5 * (z**2 + 2.0 * torch.log(std) + np.log(2.0 * np.pi))
    return torch.sum(log_probs, dim=-1)



def tangent_projected_direction_real(
    center: np.ndarray,
    gradient: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Project a Euclidean beam gradient onto the unit-sphere tangent space."""
    x = unit_norm(np.asarray(center, dtype=np.float64))
    g = np.asarray(gradient, dtype=np.float64)
    g_tangent = g - x * float(np.dot(x, g))
    nrm = float(np.linalg.norm(g_tangent))
    if nrm < eps or not np.all(np.isfinite(g_tangent)):
        return np.zeros_like(x)
    return g_tangent / nrm


def pmi_proxy_direction_dictionary_real(
    center_real: np.ndarray,
    signal_mat: np.ndarray,
    leakage_mat: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, list[str]]:
    """Beam-space directions around an SLNR center that work for arbitrary K.

    The old PMI-span policy used a two-angle coordinate system that only exists
    cleanly for K=2, because each UE had exactly one "other UE" PMI direction.
    This replacement stays in the real-packed beam space [Re{v}, Im{v}] and
    therefore works for any number of UEs.

    Directions:
      0) proxy-rate: approximately increases desired PMI gain and reduces leakage.
      1) leakage-reducing: negative tangent gradient of x^T L x.
      2) signal-increasing: positive tangent gradient of x^T G x.
    """
    x = unit_norm(center_real)
    d_leak = tangent_projected_direction_real(x, -(leakage_mat @ x), eps=eps)
    d_sig = tangent_projected_direction_real(x, signal_mat @ x, eps=eps)
    d_rate = unit_norm(d_sig + d_leak) if np.linalg.norm(d_sig + d_leak) >= eps else np.zeros_like(x)
    return np.stack([d_rate, d_leak, d_sig], axis=0), ["proxy-rate", "leakage-reducing", "signal-increasing"]


def run_wesn_policy_rl(
    cfg: SimConfig,
    rl_cfg: RLConfig,
    channels: np.ndarray,
    zf_baseline: dict[str, np.ndarray],
    slnr_baseline: dict[str, np.ndarray],
    rng: np.random.Generator,
    pmi_feedback: dict[str, np.ndarray] | None = None,
    policy_variant: str = "single_direction",
) -> dict[str, np.ndarray]:
    """Run an arbitrary-K reference-precoder-centered discrete alpha-step policy.

    This replaces the old K=2 PMI-span angle policy.  For each UE k, the policy
    starts from the selected reference beam v_k^ref and chooses a discrete alpha
    level, and optionally a direction, in the real-packed beam tangent space:

        x_k(alpha,d) = normalize(x_k^ref + alpha_step * alpha * d_k),

    where d_k is one of the PMI-only proxy directions computed from the current
    PMI vectors.  Because this construction only needs per-UE signal/leakage
    matrices, it supports any cfg.num_users.
    """
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    k = cfg.num_users
    d = 2 * cfg.num_tx_antennas
    action_dim = k
    alpha_levels_np = np.array(rl_cfg.alpha_level_grid, dtype=np.float64)
    num_alpha_levels = int(alpha_levels_np.size)
    if num_alpha_levels < 2:
        raise ValueError("alpha_level_grid must contain at least two levels.")
    zero_level_index = int(np.argmin(np.abs(alpha_levels_np)))

    if pmi_feedback is None:
        pmi_feedback = build_pmi_feedback_trace(cfg, channels)

    wesn_states = compute_wesn_states(zf_baseline["pmi_features"], rl_cfg, rng)
    state_dim = wesn_states.shape[1]

    if policy_variant not in {"single_direction", "two_direction", "multi_direction"}:
        raise ValueError("policy_variant must be one of: 'single_direction', 'two_direction', or 'multi_direction'.")

    if policy_variant == "single_direction":
        direction_names = ["proxy-rate"]
        direction_selector = [0]
    elif policy_variant == "two_direction":
        direction_names = ["leakage-reducing", "signal-increasing"]
        direction_selector = [1, 2]
    else:
        direction_names = ["proxy-rate", "leakage-reducing", "signal-increasing"]
        direction_selector = [0, 1, 2]
    num_directions = len(direction_names)
    num_action_choices = num_alpha_levels if num_directions == 1 else num_directions * num_alpha_levels

    num_logits = action_dim * num_action_choices
    w_out_init = rl_cfg.init_scale_out * rng.standard_normal((num_logits, state_dim))
    w_out = torch.nn.Parameter(torch.tensor(w_out_init, dtype=torch.float64))

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
            "The discrete alpha-level policy is implemented as exact one-sample "
            "on-policy REINFORCE. Keep --best-of-n 1 for now."
        )

    alpha_step_scale = float(rl_cfg.alpha_step_max_theta)
    uniform_alpha_entropy_per_user = float(np.log(num_alpha_levels))

    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    reward_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    advantage_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    baseline_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    kappa_trace = np.full((cfg.num_slots, k), rl_cfg.fixed_kappa, dtype=np.float64)
    beat_zf_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    beat_slnr_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    beat_baseline_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    rate_delta_baseline_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    actual_sinr_ratio_baseline_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    proxy_sinr_ratio_baseline_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
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
    fb_attempts_trace = np.ones((cfg.num_slots, k), dtype=np.float64)
    fb_accept_trace = np.ones((cfg.num_slots, k), dtype=np.float64)
    best_of_n_score_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    best_of_n_selected_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    empirical_mean_norm_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    centered_update_norm_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_entropy_trace = np.full((cfg.num_slots, k), uniform_alpha_entropy_per_user, dtype=np.float64)
    candidate_entropy_norm_trace = np.ones((cfg.num_slots, k), dtype=np.float64)
    candidate_effective_count_trace = np.ones((cfg.num_slots, k), dtype=np.float64)
    candidate_selected_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_best_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_mean_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_selected_minus_mean_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_phase_score_mean_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_phase_score_std_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_slnr_score_mean_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_slnr_score_std_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_policy_score_std_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_selected_source_trace = np.full((cfg.num_slots, k), 4.0, dtype=np.float64)
    pmi_span_selected_theta_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    pmi_span_selected_phi_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    pmi_center_theta_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    pmi_center_phi_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    pmi_mean_theta_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    pmi_mean_phi_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    learned_policy_probability_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    dk_policy_probability_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    selected_learned_branch_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    selected_dk_branch_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    mixture_logit_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    selected_alpha_level_trace = np.zeros((cfg.num_slots, action_dim), dtype=np.float64)
    mean_alpha_level_trace = np.zeros((cfg.num_slots, action_dim), dtype=np.float64)
    alpha_prob_trace = np.zeros((cfg.num_slots, action_dim, num_alpha_levels), dtype=np.float64)
    selected_direction_trace = np.zeros((cfg.num_slots, action_dim), dtype=np.int64)
    direction_prob_trace = np.zeros((cfg.num_slots, action_dim, num_directions), dtype=np.float64)
    batch_s: list[np.ndarray] = []
    batch_alpha_indices: list[np.ndarray] = []
    batch_branch_is_learned: list[float] = []
    batch_adv: list[float] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        nonlocal batch_s, batch_alpha_indices, batch_branch_is_learned, batch_adv, batch_indices
        if not batch_s:
            return

        s_batch = torch.tensor(np.stack(batch_s, axis=0), dtype=torch.float64)
        alpha_index_batch = torch.tensor(np.stack(batch_alpha_indices, axis=0), dtype=torch.long)
        branch_is_learned_batch = torch.tensor(np.array(batch_branch_is_learned), dtype=torch.float64)
        adv_batch = torch.tensor(np.array(batch_adv), dtype=torch.float64)

        choice_logits_batch = torch.einsum("ln,bn->bl", w_out, s_batch).reshape(
            -1, action_dim, num_action_choices
        )
        if rl_cfg.alpha_zero_logit_bias != 0.0:
            zero_prior = torch.zeros_like(choice_logits_batch)
            if num_directions == 1:
                zero_prior[:, :, zero_level_index] = float(rl_cfg.alpha_zero_logit_bias)
            else:
                for dir_idx in range(num_directions):
                    zero_prior[:, :, dir_idx * num_alpha_levels + zero_level_index] = float(rl_cfg.alpha_zero_logit_bias)
            choice_logits_batch = choice_logits_batch + zero_prior
        choice_log_probs_batch = torch.log_softmax(
            choice_logits_batch / max(float(rl_cfg.candidate_temperature), 1e-12), dim=-1
        )
        learned_branch_log_prob = torch.gather(
            choice_log_probs_batch, dim=-1, index=alpha_index_batch.unsqueeze(-1)
        ).squeeze(-1).sum(dim=-1)

        p_learned = torch.sigmoid(mixture_logit)
        log_p_learned = torch.log(torch.clamp(p_learned, min=1e-12))
        log_p_dk = torch.log(torch.clamp(1.0 - p_learned, min=1e-12))
        branch_log_prob = branch_is_learned_batch * log_p_learned + (1.0 - branch_is_learned_batch) * log_p_dk
        joint_log_prob = branch_log_prob + branch_is_learned_batch * learned_branch_log_prob
        rl_loss = -torch.mean(adv_batch * joint_log_prob)
        loss = rl_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([w_out, mixture_logit], rl_cfg.grad_clip_norm)
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        rl_loss_value = float(rl_loss.detach().cpu().item())
        grad_norm_value = float(grad_norm.detach().cpu().item())
        for idx in batch_indices:
            loss_trace[idx] = loss_value
            rl_loss_trace[idx] = rl_loss_value
            joint_pmi_aux_loss_trace[idx] = 0.0
            grad_norm_trace[idx] = grad_norm_value

        batch_s = []
        batch_alpha_indices = []
        batch_branch_is_learned = []
        batch_adv = []
        batch_indices = []

    for t in range(cfg.num_slots):
        print(f"WESN-RL Slot {t + 1} / {cfg.num_slots}", end="\r")
        s = wesn_states[t]
        s_t = torch.tensor(s, dtype=torch.float64)

        q_vectors = pmi_feedback["q_vectors"][t]
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

        if rl_cfg.reference_precoder == "slnr":
            ref_precoder_t = slnr_precoder_t
            ref_rate = slnr_rate
            ref_actual_sinr = slnr_actual_sinr
            ref_proxy_sinr = slnr_proxy_sinr
            ref_name = "SLNR"
        elif rl_cfg.reference_precoder == "zf":
            ref_precoder_t = zf_baseline["precoders"][t]
            ref_rate = zf_rate
            ref_actual_sinr = zf_actual_sinr
            ref_proxy_sinr = zf_proxy_sinr
            ref_name = "ZF"
        else:
            raise ValueError(f"Unknown reference_precoder={rl_cfg.reference_precoder!r}")

        center_real = np.stack([complex_unit_to_real(ref_precoder_t[:, ku]) for ku in range(k)], axis=0)
        all_dirs = np.zeros((k, 3, d), dtype=np.float64)
        for ku in range(k):
            dirs_ku, _ = pmi_proxy_direction_dictionary_real(
                center_real=center_real[ku],
                signal_mat=signal_mats[ku],
                leakage_mat=leakage_mats[ku],
                eps=rl_cfg.leakage_norm_eps,
            )
            all_dirs[ku] = dirs_ku
        angle_directions_np = all_dirs[:, direction_selector, :]

        with torch.no_grad():
            choice_logits_t = torch.einsum("ln,n->l", w_out, s_t).reshape(action_dim, num_action_choices)
            if rl_cfg.alpha_zero_logit_bias != 0.0:
                if num_directions == 1:
                    choice_logits_t[:, zero_level_index] += float(rl_cfg.alpha_zero_logit_bias)
                else:
                    for dir_idx in range(num_directions):
                        choice_logits_t[:, dir_idx * num_alpha_levels + zero_level_index] += float(rl_cfg.alpha_zero_logit_bias)
            choice_log_probs_t = torch.log_softmax(
                choice_logits_t / max(float(rl_cfg.candidate_temperature), 1e-12), dim=-1
            )
            choice_probs_np = torch.exp(choice_log_probs_t).detach().cpu().numpy()
            choice_logits_np = choice_logits_t.detach().cpu().numpy()
            if num_directions == 1:
                alpha_probs_np = choice_probs_np
                direction_probs_np = np.ones((action_dim, 1), dtype=np.float64)
                alpha_mean_level_np = alpha_probs_np @ alpha_levels_np
            else:
                pair_probs_np = choice_probs_np.reshape(action_dim, num_directions, num_alpha_levels)
                alpha_probs_np = pair_probs_np.sum(axis=1)
                direction_probs_np = pair_probs_np.sum(axis=2)
                alpha_mean_level_np = alpha_probs_np @ alpha_levels_np

        with torch.no_grad():
            p_learned_np = float(torch.sigmoid(mixture_logit).detach().cpu().item())
        slot_from_learned = bool(rng.uniform(0.0, 1.0) < p_learned_np)
        learned_policy_probability_trace[t] = p_learned_np
        dk_policy_probability_trace[t] = 1.0 - p_learned_np
        selected_learned_branch_trace[t] = 1.0 if slot_from_learned else 0.0
        selected_dk_branch_trace[t] = 1.0 - selected_learned_branch_trace[t]
        mixture_logit_trace[t] = float(mixture_logit.detach().cpu().item())

        if slot_from_learned:
            selected_choice_indices = np.array([
                int(rng.choice(num_action_choices, p=choice_probs_np[i]))
                for i in range(action_dim)
            ], dtype=np.int64)
            if num_directions == 1:
                selected_direction_indices = np.zeros(action_dim, dtype=np.int64)
                selected_alpha_indices = selected_choice_indices.copy()
            else:
                selected_direction_indices = selected_choice_indices // num_alpha_levels
                selected_alpha_indices = selected_choice_indices % num_alpha_levels
            selected_alpha_levels = alpha_levels_np[selected_alpha_indices]

            beams = np.zeros((cfg.num_tx_antennas, k), dtype=np.complex128)
            x_sample = np.zeros((k, d), dtype=np.float64)
            mean_x_sample = np.zeros((k, d), dtype=np.float64)
            signal = np.zeros(k, dtype=np.float64)
            leakage = np.zeros(k, dtype=np.float64)
            for ku in range(k):
                selected_dir = angle_directions_np[ku, selected_direction_indices[ku]]
                x_sample[ku] = unit_norm(center_real[ku] + alpha_step_scale * selected_alpha_levels[ku] * selected_dir)

                if num_directions == 1:
                    mean_dir_step = alpha_step_scale * alpha_mean_level_np[ku] * angle_directions_np[ku, 0]
                else:
                    mean_dir_step = np.zeros(d, dtype=np.float64)
                    pair_probs_i = choice_probs_np[ku].reshape(num_directions, num_alpha_levels)
                    for dir_idx in range(num_directions):
                        for alpha_idx, alpha_level in enumerate(alpha_levels_np):
                            mean_dir_step += pair_probs_i[dir_idx, alpha_idx] * alpha_step_scale * alpha_level * angle_directions_np[ku, dir_idx]
                mean_x_sample[ku] = unit_norm(center_real[ku] + mean_dir_step)
                beams[:, ku] = real_to_complex_beam(x_sample[ku], cfg.total_tx_power, k)
                signal[ku] = raw_quadratic_score(x_sample[ku], signal_mats[ku])
                leakage[ku] = raw_quadratic_score(x_sample[ku], leakage_mats[ku])

            rate, actual_sinr = compute_slot_sum_rate(channels[t], beams, noise_power)
            proxy_sinr = compute_pmi_sinr_proxy_from_precoder(
                q_vectors, beams, noise_power, eps=rl_cfg.reward_sinr_eps
            )
            reward_ref_rate = compute_reward_reference_rate(
                reward_mode=rl_cfg.reward_mode,
                continuous_reference_rate=ref_rate,
                reference_actual_sinr=ref_actual_sinr,
                eps=rl_cfg.reward_sinr_eps,
            )
            base_reward = compute_rl_reward(
                reward_mode=rl_cfg.reward_mode,
                actual_sinr=actual_sinr,
                reference_actual_sinr=ref_actual_sinr,
                eps=rl_cfg.reward_sinr_eps,
            )
            reward = add_positive_rate_delta_bonus(
                base_reward=base_reward,
                rate=rate,
                ref_rate=reward_ref_rate,
                bonus_lambda=rl_cfg.positive_rate_bonus_lambda,
                bonus_power=rl_cfg.positive_rate_bonus_power,
            )
            emp_mean_sample = mean_x_sample
            best_score = reward
            alpha_indices_for_training = selected_choice_indices
            selected_alpha_for_trace = selected_alpha_levels
            mean_alpha_for_trace = alpha_mean_level_np
            selected_direction_for_trace = selected_direction_indices
        else:
            beams = ref_precoder_t.copy()
            x_sample = center_real.copy()
            signal = np.zeros(k, dtype=np.float64)
            leakage = np.zeros(k, dtype=np.float64)
            for ku in range(k):
                signal[ku] = raw_quadratic_score(x_sample[ku], signal_mats[ku])
                leakage[ku] = raw_quadratic_score(x_sample[ku], leakage_mats[ku])
            rate = ref_rate
            actual_sinr = ref_actual_sinr
            proxy_sinr = ref_proxy_sinr
            reward_ref_rate = compute_reward_reference_rate(
                reward_mode=rl_cfg.reward_mode,
                continuous_reference_rate=ref_rate,
                reference_actual_sinr=ref_actual_sinr,
                eps=rl_cfg.reward_sinr_eps,
            )
            base_reward = compute_rl_reward(
                reward_mode=rl_cfg.reward_mode,
                actual_sinr=actual_sinr,
                reference_actual_sinr=ref_actual_sinr,
                eps=rl_cfg.reward_sinr_eps,
            )
            reward = add_positive_rate_delta_bonus(
                base_reward=base_reward,
                rate=rate,
                ref_rate=reward_ref_rate,
                bonus_lambda=rl_cfg.positive_rate_bonus_lambda,
                bonus_power=rl_cfg.positive_rate_bonus_power,
            )
            emp_mean_sample = x_sample.copy()
            best_score = reward
            alpha_indices_for_training = np.full(action_dim, zero_level_index, dtype=np.int64)
            selected_alpha_for_trace = np.zeros(action_dim, dtype=np.float64)
            mean_alpha_for_trace = alpha_mean_level_np
            selected_direction_for_trace = np.zeros(action_dim, dtype=np.int64)

        selected_alpha_level_trace[t] = np.asarray(selected_alpha_for_trace, dtype=np.float64).reshape(-1)
        mean_alpha_level_trace[t] = np.asarray(mean_alpha_for_trace, dtype=np.float64).reshape(-1)
        alpha_prob_trace[t] = alpha_probs_np
        selected_direction_trace[t] = np.asarray(selected_direction_for_trace, dtype=np.int64).reshape(-1)
        direction_prob_trace[t] = direction_probs_np

        signal_trace[t] = signal
        leakage_trace[t] = leakage
        empirical_mean_norm_trace[t] = np.linalg.norm(emp_mean_sample, axis=1)
        centered_update_norm_trace[t] = np.linalg.norm(x_sample - emp_mean_sample, axis=1)
        candidate_selected_score_trace[t] = selected_alpha_for_trace
        candidate_mean_score_trace[t] = mean_alpha_for_trace
        candidate_selected_minus_mean_score_trace[t] = selected_alpha_for_trace - mean_alpha_for_trace
        candidate_phase_score_mean_trace[t] = mean_alpha_for_trace
        candidate_phase_score_std_trace[t] = selected_alpha_for_trace
        candidate_slnr_score_mean_trace[t] = mean_alpha_for_trace
        candidate_slnr_score_std_trace[t] = selected_alpha_for_trace

        ent_per_user = -np.sum(alpha_probs_np * np.log(np.maximum(alpha_probs_np, 1e-12)), axis=1)
        eff_per_user = 1.0 / np.maximum(np.sum(alpha_probs_np**2, axis=1), 1e-12)
        candidate_entropy_trace[t] = ent_per_user
        candidate_entropy_norm_trace[t] = ent_per_user / max(np.log(num_alpha_levels), 1e-12)
        candidate_effective_count_trace[t] = eff_per_user
        candidate_policy_score_std_trace[t] = np.std(choice_logits_np.reshape(k, num_action_choices), axis=-1)

        advantage = float(reward)
        if rl_cfg.advantage_clip > 0:
            advantage = float(np.clip(advantage, -rl_cfg.advantage_clip, rl_cfg.advantage_clip))
        reward_baseline_value = 0.0

        throughput[t] = float(rate)
        reward_trace[t] = float(reward)
        rate_delta_trace[t] = float(rate) - zf_rate
        rate_delta_baseline_trace[t] = float(rate) - ref_rate
        proxy_sinr_trace[t] = proxy_sinr
        zf_proxy_sinr_trace[t] = zf_proxy_sinr
        proxy_sinr_ratio_trace[t] = proxy_sinr / np.maximum(zf_proxy_sinr, rl_cfg.reward_sinr_eps)
        actual_sinr_ratio_trace[t] = actual_sinr / np.maximum(zf_actual_sinr, rl_cfg.reward_sinr_eps)
        proxy_sinr_ratio_baseline_trace[t] = proxy_sinr / np.maximum(ref_proxy_sinr, rl_cfg.reward_sinr_eps)
        actual_sinr_ratio_baseline_trace[t] = actual_sinr / np.maximum(ref_actual_sinr, rl_cfg.reward_sinr_eps)
        advantage_trace[t] = advantage
        baseline_trace[t] = reward_baseline_value
        beat_zf_trace[t] = 1.0 if rate > zf_rate else 0.0
        beat_slnr_trace[t] = 1.0 if rate > slnr_rate else 0.0
        beat_baseline_trace[t] = 1.0 if rate > ref_rate else 0.0
        beam_similarity_trace[t] = beam_similarity(beams, zf_baseline["precoders"][t])
        best_of_n_score_trace[t] = best_score
        best_of_n_selected_trace[t] = 0.0

        batch_s.append(s)
        batch_alpha_indices.append(alpha_indices_for_training)
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
        "rate_delta_baseline": rate_delta_baseline_trace,
        "rate_delta_slnr": rate_delta_baseline_trace if rl_cfg.reference_precoder == "slnr" else throughput - slnr_baseline["throughput"],
        "proxy_sinr": proxy_sinr_trace,
        "zf_proxy_sinr": zf_proxy_sinr_trace,
        "proxy_sinr_ratio": proxy_sinr_ratio_trace,
        "actual_sinr_ratio": actual_sinr_ratio_trace,
        "proxy_sinr_ratio_baseline": proxy_sinr_ratio_baseline_trace,
        "proxy_sinr_ratio_slnr": proxy_sinr_ratio_baseline_trace if rl_cfg.reference_precoder == "slnr" else proxy_sinr_trace / np.maximum(
            np.stack([compute_pmi_sinr_proxy_from_precoder(pmi_feedback["q_vectors"][tt], slnr_baseline["precoders"][tt], noise_power, eps=rl_cfg.reward_sinr_eps) for tt in range(cfg.num_slots)], axis=0),
            rl_cfg.reward_sinr_eps,
        ),
        "actual_sinr_ratio_baseline": actual_sinr_ratio_baseline_trace,
        "actual_sinr_ratio_slnr": actual_sinr_ratio_baseline_trace if rl_cfg.reference_precoder == "slnr" else (actual_sinr_ratio_trace * zf_baseline["sinr"]) / np.maximum(slnr_baseline["sinr"], rl_cfg.reward_sinr_eps),
        "advantage": advantage_trace,
        "reward_baseline": baseline_trace,
        "kappa": kappa_trace,
        "beat_zf": beat_zf_trace,
        "beat_slnr": beat_slnr_trace,
        "beat_baseline": beat_baseline_trace,
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
        "pmi_span_selected_theta": pmi_span_selected_theta_trace,
        "pmi_span_selected_phi": pmi_span_selected_phi_trace,
        "pmi_center_theta": pmi_center_theta_trace,
        "pmi_center_phi": pmi_center_phi_trace,
        "pmi_mean_theta": pmi_mean_theta_trace,
        "pmi_mean_phi": pmi_mean_phi_trace,
        "selected_alpha_level": selected_alpha_level_trace,
        "mean_alpha_level": mean_alpha_level_trace,
        "alpha_prob": alpha_prob_trace,
        "selected_direction": selected_direction_trace,
        "direction_prob": direction_prob_trace,
        "direction_names": np.array(direction_names, dtype=object),
        "alpha_levels": alpha_levels_np,
        "policy_variant": policy_variant,
        "wesn_states": wesn_states,
    }

def run_wesn_policy_rl_multi_iteration(
    cfg: SimConfig,
    rl_cfg: RLConfig,
    channels: np.ndarray,
    zf_baseline: dict[str, np.ndarray],
    slnr_baseline: dict[str, np.ndarray],
    rng: np.random.Generator,
    pmi_feedback: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Run a sequential multi-iteration WESN perturbation policy with stop actions.

    This policy turns each channel slot into a short deterministic inner MDP.
    The initial inner state is the selected reference precoder.  At each inner
    step h, the same WESN readout is evaluated on an augmented state containing
    the fixed WESN PMI-memory state, the current real-packed beams, the beam
    displacement from the reference, current PMI signal/leakage metrics, active
    UE flags, and h/H_max.  Each active UE chooses either STOP or one
    (direction, alpha) perturbation.  The final true-channel rate is used as one
    terminal REINFORCE reward for the whole inner trajectory.
    """
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    k = cfg.num_users
    d = 2 * cfg.num_tx_antennas
    action_dim = k
    h_max = max(1, int(rl_cfg.multi_iter_max_steps))
    include_stop = bool(rl_cfg.multi_iter_include_stop_action)

    alpha_levels_np = np.array(rl_cfg.alpha_level_grid, dtype=np.float64)
    num_alpha_levels = int(alpha_levels_np.size)
    if num_alpha_levels < 2:
        raise ValueError("alpha_level_grid must contain at least two levels.")
    zero_level_index = int(np.argmin(np.abs(alpha_levels_np)))

    if pmi_feedback is None:
        pmi_feedback = build_pmi_feedback_trace(cfg, channels)

    wesn_states = compute_wesn_states(zf_baseline["pmi_features"], rl_cfg, rng)
    base_state_dim = wesn_states.shape[1]

    direction_names = ["proxy-rate", "leakage-reducing", "signal-increasing"]
    direction_selector = [0, 1, 2]
    num_directions = len(direction_names)
    num_action_choices = num_directions * num_alpha_levels + (1 if include_stop else 0)
    stop_choice_index = 0 if include_stop else -1
    first_move_choice_index = 1 if include_stop else 0

    # Augmented state = WESN PMI memory + current beams + displacement from
    # reference + current PMI signal/leakage + active flags + normalized h.
    aug_state_dim = base_state_dim + 2 * k * d + 2 * k + k + 1
    num_logits = action_dim * num_action_choices
    w_out_init = rl_cfg.init_scale_out * rng.standard_normal((num_logits, aug_state_dim))
    w_out = torch.nn.Parameter(torch.tensor(w_out_init, dtype=torch.float64))
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

    alpha_step_scale = float(rl_cfg.alpha_step_max_theta)

    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    reward_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    advantage_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    baseline_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    kappa_trace = np.full((cfg.num_slots, k), rl_cfg.fixed_kappa, dtype=np.float64)
    beat_zf_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    beat_slnr_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    beat_baseline_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    rate_delta_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    rate_delta_baseline_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    proxy_sinr_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    zf_proxy_sinr_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    proxy_sinr_ratio_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    actual_sinr_ratio_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    proxy_sinr_ratio_baseline_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    actual_sinr_ratio_baseline_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    beam_similarity_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    grad_norm_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    loss_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    rl_loss_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    joint_pmi_aux_loss_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    signal_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    leakage_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    fb_attempts_trace = np.ones((cfg.num_slots, k), dtype=np.float64)
    fb_accept_trace = np.ones((cfg.num_slots, k), dtype=np.float64)
    best_of_n_score_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    best_of_n_selected_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    empirical_mean_norm_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    centered_update_norm_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_entropy_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_entropy_norm_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_effective_count_trace = np.ones((cfg.num_slots, k), dtype=np.float64)
    candidate_selected_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_best_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_mean_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_selected_minus_mean_score_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_phase_score_mean_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_phase_score_std_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_slnr_score_mean_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_slnr_score_std_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_policy_score_std_trace = np.zeros((cfg.num_slots, k), dtype=np.float64)
    candidate_selected_source_trace = np.full((cfg.num_slots, k), 5.0, dtype=np.float64)
    learned_policy_probability_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    dk_policy_probability_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    selected_learned_branch_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    selected_dk_branch_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    mixture_logit_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    selected_alpha_level_trace = np.zeros((cfg.num_slots, action_dim), dtype=np.float64)
    mean_alpha_level_trace = np.zeros((cfg.num_slots, action_dim), dtype=np.float64)
    alpha_prob_trace = np.zeros((cfg.num_slots, action_dim, num_alpha_levels), dtype=np.float64)
    selected_direction_trace = np.zeros((cfg.num_slots, action_dim), dtype=np.int64)
    direction_prob_trace = np.zeros((cfg.num_slots, action_dim, num_directions), dtype=np.float64)
    stop_prob_trace = np.zeros((cfg.num_slots, action_dim), dtype=np.float64)
    selected_stop_trace = np.zeros((cfg.num_slots, action_dim), dtype=np.float64)
    num_inner_steps_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    reference_like_rate_baseline_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    reference_like_precoder_distance_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    reference_like_baseline_update_weight_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    reference_like_rate_baseline: float | None = None

    batch_traj_states: list[np.ndarray] = []
    batch_traj_choices: list[np.ndarray] = []
    batch_traj_masks: list[np.ndarray] = []
    batch_branch_is_learned: list[float] = []
    batch_adv: list[float] = []
    batch_indices: list[int] = []

    def add_choice_prior(logits: torch.Tensor) -> torch.Tensor:
        if rl_cfg.alpha_zero_logit_bias == 0.0:
            return logits
        prior = torch.zeros_like(logits)
        if include_stop:
            prior[..., stop_choice_index] = float(rl_cfg.alpha_zero_logit_bias)
        for dir_idx in range(num_directions):
            zero_choice = first_move_choice_index + dir_idx * num_alpha_levels + zero_level_index
            prior[..., zero_choice] = float(rl_cfg.alpha_zero_logit_bias)
        return logits + prior

    def build_aug_state(
        base_s: np.ndarray,
        x_current: np.ndarray,
        x_ref: np.ndarray,
        signal_now: np.ndarray,
        leakage_now: np.ndarray,
        active_mask: np.ndarray,
        h_idx: int,
    ) -> np.ndarray:
        return np.concatenate(
            [
                base_s,
                x_current.reshape(-1),
                (x_current - x_ref).reshape(-1),
                signal_now.reshape(-1),
                leakage_now.reshape(-1),
                active_mask.astype(np.float64).reshape(-1),
                np.array([float(h_idx) / max(float(h_max), 1.0)], dtype=np.float64),
            ]
        ).astype(np.float64)

    def decode_move_choice(choice: int) -> tuple[int, int]:
        move = int(choice) - first_move_choice_index
        dir_idx = move // num_alpha_levels
        alpha_idx = move % num_alpha_levels
        return int(dir_idx), int(alpha_idx)

    def flush_batch() -> None:
        nonlocal batch_traj_states, batch_traj_choices, batch_traj_masks, batch_branch_is_learned, batch_adv, batch_indices
        if not batch_adv:
            return

        p_learned = torch.sigmoid(mixture_logit)
        log_p_learned = torch.log(torch.clamp(p_learned, min=1e-12))
        log_p_dk = torch.log(torch.clamp(1.0 - p_learned, min=1e-12))
        joint_log_probs: list[torch.Tensor] = []

        for states_np, choices_np, masks_np, branch in zip(
            batch_traj_states,
            batch_traj_choices,
            batch_traj_masks,
            batch_branch_is_learned,
        ):
            branch_log_prob = float(branch) * log_p_learned + (1.0 - float(branch)) * log_p_dk
            if float(branch) <= 0.5 or states_np.size == 0:
                joint_log_probs.append(branch_log_prob)
                continue

            states_t = torch.tensor(states_np, dtype=torch.float64)
            choices_t = torch.tensor(choices_np, dtype=torch.long)
            masks_t = torch.tensor(masks_np, dtype=torch.float64)
            logits_t = torch.einsum("ln,bn->bl", w_out, states_t).reshape(
                -1, action_dim, num_action_choices
            )
            logits_t = add_choice_prior(logits_t)
            log_probs_t = torch.log_softmax(
                logits_t / max(float(rl_cfg.candidate_temperature), 1e-12), dim=-1
            )
            selected_log_probs = torch.gather(
                log_probs_t, dim=-1, index=choices_t.unsqueeze(-1)
            ).squeeze(-1)
            learned_log_prob = torch.sum(selected_log_probs * masks_t)
            joint_log_probs.append(branch_log_prob + learned_log_prob)

        joint_log_prob_batch = torch.stack(joint_log_probs)
        adv_batch = torch.tensor(np.array(batch_adv), dtype=torch.float64)
        rl_loss = -torch.mean(adv_batch * joint_log_prob_batch)
        loss = rl_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([w_out, mixture_logit], rl_cfg.grad_clip_norm)
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        rl_loss_value = float(rl_loss.detach().cpu().item())
        grad_norm_value = float(grad_norm.detach().cpu().item())
        for idx in batch_indices:
            loss_trace[idx] = loss_value
            rl_loss_trace[idx] = rl_loss_value
            joint_pmi_aux_loss_trace[idx] = 0.0
            grad_norm_trace[idx] = grad_norm_value

        batch_traj_states = []
        batch_traj_choices = []
        batch_traj_masks = []
        batch_branch_is_learned = []
        batch_adv = []
        batch_indices = []

    for t in range(cfg.num_slots):
        print(f"WESN-RL multi-iteration Slot {t + 1} / {cfg.num_slots}", end="\r")
        base_s = wesn_states[t]
        q_vectors = pmi_feedback["q_vectors"][t]
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

        if rl_cfg.reference_precoder == "slnr":
            ref_precoder_t = slnr_precoder_t
            ref_rate = slnr_rate
            ref_actual_sinr = slnr_actual_sinr
            ref_proxy_sinr = slnr_proxy_sinr
        elif rl_cfg.reference_precoder == "zf":
            ref_precoder_t = zf_baseline["precoders"][t]
            ref_rate = zf_rate
            ref_actual_sinr = zf_actual_sinr
            ref_proxy_sinr = zf_proxy_sinr
        else:
            raise ValueError(f"Unknown reference_precoder={rl_cfg.reference_precoder!r}")

        center_real = np.stack([complex_unit_to_real(ref_precoder_t[:, ku]) for ku in range(k)], axis=0)
        x_current = center_real.copy()
        active_mask = np.ones(k, dtype=bool)
        traj_states: list[np.ndarray] = []
        traj_choices: list[np.ndarray] = []
        traj_masks: list[np.ndarray] = []
        last_choice_probs = np.zeros((action_dim, num_action_choices), dtype=np.float64)
        last_alpha_probs = np.zeros((action_dim, num_alpha_levels), dtype=np.float64)
        last_direction_probs = np.zeros((action_dim, num_directions), dtype=np.float64)
        last_stop_probs = np.zeros(action_dim, dtype=np.float64)
        selected_alpha_last = np.zeros(action_dim, dtype=np.float64)
        selected_direction_last = np.zeros(action_dim, dtype=np.int64)
        selected_stop_any = np.zeros(action_dim, dtype=np.float64)
        steps_executed = 0

        with torch.no_grad():
            p_learned_np = float(torch.sigmoid(mixture_logit).detach().cpu().item())
        slot_from_learned = bool(rng.uniform(0.0, 1.0) < p_learned_np)
        learned_policy_probability_trace[t] = p_learned_np
        dk_policy_probability_trace[t] = 1.0 - p_learned_np
        selected_learned_branch_trace[t] = 1.0 if slot_from_learned else 0.0
        selected_dk_branch_trace[t] = 1.0 - selected_learned_branch_trace[t]
        mixture_logit_trace[t] = float(mixture_logit.detach().cpu().item())

        if slot_from_learned:
            for h in range(h_max):
                signal_now = np.array([
                    raw_quadratic_score(x_current[ku], signal_mats[ku]) for ku in range(k)
                ], dtype=np.float64)
                leakage_now = np.array([
                    raw_quadratic_score(x_current[ku], leakage_mats[ku]) for ku in range(k)
                ], dtype=np.float64)
                aug_state = build_aug_state(
                    base_s=base_s,
                    x_current=x_current,
                    x_ref=center_real,
                    signal_now=signal_now,
                    leakage_now=leakage_now,
                    active_mask=active_mask,
                    h_idx=h,
                )

                with torch.no_grad():
                    aug_t = torch.tensor(aug_state, dtype=torch.float64)
                    logits_t = torch.einsum("ln,n->l", w_out, aug_t).reshape(action_dim, num_action_choices)
                    logits_t = add_choice_prior(logits_t)
                    log_probs_t = torch.log_softmax(
                        logits_t / max(float(rl_cfg.candidate_temperature), 1e-12), dim=-1
                    )
                    choice_probs_np = torch.exp(log_probs_t).detach().cpu().numpy()

                choices = np.zeros(action_dim, dtype=np.int64)
                decision_mask = active_mask.astype(np.float64)
                if include_stop:
                    last_stop_probs = choice_probs_np[:, stop_choice_index]
                    move_probs = choice_probs_np[:, first_move_choice_index:].reshape(
                        action_dim, num_directions, num_alpha_levels
                    )
                else:
                    move_probs = choice_probs_np.reshape(action_dim, num_directions, num_alpha_levels)
                last_alpha_probs = move_probs.sum(axis=1)
                last_direction_probs = move_probs.sum(axis=2)
                last_choice_probs = choice_probs_np.copy()

                all_dirs = np.zeros((k, num_directions, d), dtype=np.float64)
                for ku in range(k):
                    dirs_ku, _ = pmi_proxy_direction_dictionary_real(
                        center_real=x_current[ku],
                        signal_mat=signal_mats[ku],
                        leakage_mat=leakage_mats[ku],
                        eps=rl_cfg.leakage_norm_eps,
                    )
                    all_dirs[ku] = dirs_ku[direction_selector]

                for ku in range(k):
                    if not active_mask[ku]:
                        choices[ku] = stop_choice_index if include_stop else first_move_choice_index + zero_level_index
                        continue
                    choices[ku] = int(rng.choice(num_action_choices, p=choice_probs_np[ku]))
                    if include_stop and choices[ku] == stop_choice_index:
                        active_mask[ku] = False
                        selected_stop_any[ku] = 1.0
                        continue
                    dir_idx, alpha_idx = decode_move_choice(choices[ku])
                    alpha_level = float(alpha_levels_np[alpha_idx])
                    selected_direction_last[ku] = dir_idx
                    selected_alpha_last[ku] = alpha_level
                    x_current[ku] = unit_norm(
                        x_current[ku] + alpha_step_scale * alpha_level * all_dirs[ku, dir_idx]
                    )

                traj_states.append(aug_state)
                traj_choices.append(choices)
                traj_masks.append(decision_mask)
                steps_executed = h + 1
                if include_stop and not np.any(active_mask):
                    break
        else:
            x_current = center_real.copy()
            steps_executed = 0

        beams = np.zeros((cfg.num_tx_antennas, k), dtype=np.complex128)
        for ku in range(k):
            beams[:, ku] = real_to_complex_beam(x_current[ku], cfg.total_tx_power, k)
        signal = np.array([raw_quadratic_score(x_current[ku], signal_mats[ku]) for ku in range(k)], dtype=np.float64)
        leakage = np.array([raw_quadratic_score(x_current[ku], leakage_mats[ku]) for ku in range(k)], dtype=np.float64)

        if slot_from_learned:
            rate, actual_sinr = compute_slot_sum_rate(channels[t], beams, noise_power)
            proxy_sinr = compute_pmi_sinr_proxy_from_precoder(
                q_vectors, beams, noise_power, eps=rl_cfg.reward_sinr_eps
            )
        else:
            rate = ref_rate
            actual_sinr = ref_actual_sinr
            proxy_sinr = ref_proxy_sinr

        reward_ref_rate = compute_reward_reference_rate(
            reward_mode=rl_cfg.reward_mode,
            continuous_reference_rate=ref_rate,
            reference_actual_sinr=ref_actual_sinr,
            eps=rl_cfg.reward_sinr_eps,
        )
        base_reward = compute_rl_reward(
            reward_mode=rl_cfg.reward_mode,
            actual_sinr=actual_sinr,
            reference_actual_sinr=ref_actual_sinr,
            eps=rl_cfg.reward_sinr_eps,
        )
        reward = add_positive_rate_delta_bonus(
            base_reward=base_reward,
            rate=rate,
            ref_rate=reward_ref_rate,
            bonus_lambda=rl_cfg.positive_rate_bonus_lambda,
            bonus_power=rl_cfg.positive_rate_bonus_power,
        )
        if slot_from_learned and rl_cfg.multi_iter_step_penalty != 0.0:
            reward = float(reward - float(rl_cfg.multi_iter_step_penalty) * float(steps_executed))

        advantage = float(reward)
        if rl_cfg.advantage_clip > 0:
            advantage = float(np.clip(advantage, -rl_cfg.advantage_clip, rl_cfg.advantage_clip))

        throughput[t] = float(rate)
        reward_trace[t] = float(reward)
        rate_delta_trace[t] = float(rate) - zf_rate
        rate_delta_baseline_trace[t] = float(rate) - ref_rate
        proxy_sinr_trace[t] = proxy_sinr
        zf_proxy_sinr_trace[t] = zf_proxy_sinr
        proxy_sinr_ratio_trace[t] = proxy_sinr / np.maximum(zf_proxy_sinr, rl_cfg.reward_sinr_eps)
        actual_sinr_ratio_trace[t] = actual_sinr / np.maximum(zf_actual_sinr, rl_cfg.reward_sinr_eps)
        proxy_sinr_ratio_baseline_trace[t] = proxy_sinr / np.maximum(ref_proxy_sinr, rl_cfg.reward_sinr_eps)
        actual_sinr_ratio_baseline_trace[t] = actual_sinr / np.maximum(ref_actual_sinr, rl_cfg.reward_sinr_eps)
        advantage_trace[t] = advantage
        baseline_trace[t] = 0.0
        beat_zf_trace[t] = 1.0 if rate > zf_rate else 0.0
        beat_slnr_trace[t] = 1.0 if rate > slnr_rate else 0.0
        beat_baseline_trace[t] = 1.0 if rate > ref_rate else 0.0
        beam_similarity_trace[t] = beam_similarity(beams, zf_baseline["precoders"][t])
        best_of_n_score_trace[t] = reward
        best_of_n_selected_trace[t] = 0.0

        ref_like_distance = phase_aligned_precoder_distance(
            reference_precoder=ref_precoder_t,
            learned_precoder=beams,
            eps=rl_cfg.reference_like_baseline_eps,
        )
        ref_like_update_weight = reference_like_baseline_update_weight(
            precoder_distance=ref_like_distance,
            alpha_max=rl_cfg.reference_like_baseline_alpha_max,
            sigma_d=rl_cfg.reference_like_baseline_sigma_d,
            eps=rl_cfg.reference_like_baseline_eps,
        )
        cqi_quantized_rate = compute_cqi_quantized_sum_rate_from_sinr(
            actual_sinr, eps=rl_cfg.reward_sinr_eps
        )
        if reference_like_rate_baseline is None:
            reference_like_rate_baseline = cqi_quantized_rate
        reference_like_rate_baseline_trace[t] = float(reference_like_rate_baseline)
        reference_like_precoder_distance_trace[t] = ref_like_distance
        reference_like_baseline_update_weight_trace[t] = ref_like_update_weight
        reference_like_rate_baseline = (
            (1.0 - ref_like_update_weight) * float(reference_like_rate_baseline)
            + ref_like_update_weight * cqi_quantized_rate
        )

        signal_trace[t] = signal
        leakage_trace[t] = leakage
        empirical_mean_norm_trace[t] = np.linalg.norm(x_current, axis=1)
        centered_update_norm_trace[t] = np.linalg.norm(x_current - center_real, axis=1)
        selected_alpha_level_trace[t] = selected_alpha_last
        mean_alpha_level_trace[t] = last_alpha_probs @ alpha_levels_np if last_alpha_probs.size else 0.0
        alpha_prob_trace[t] = last_alpha_probs
        selected_direction_trace[t] = selected_direction_last
        direction_prob_trace[t] = last_direction_probs
        stop_prob_trace[t] = last_stop_probs
        selected_stop_trace[t] = selected_stop_any
        num_inner_steps_trace[t] = float(steps_executed)
        candidate_selected_score_trace[t] = selected_alpha_last
        candidate_mean_score_trace[t] = mean_alpha_level_trace[t]
        candidate_selected_minus_mean_score_trace[t] = selected_alpha_last - mean_alpha_level_trace[t]
        candidate_phase_score_mean_trace[t] = mean_alpha_level_trace[t]
        candidate_phase_score_std_trace[t] = selected_alpha_last
        candidate_slnr_score_mean_trace[t] = mean_alpha_level_trace[t]
        candidate_slnr_score_std_trace[t] = selected_alpha_last
        if last_choice_probs.size:
            ent = -np.sum(last_choice_probs * np.log(np.maximum(last_choice_probs, 1e-12)), axis=1)
            candidate_entropy_trace[t] = ent
            candidate_entropy_norm_trace[t] = ent / max(np.log(num_action_choices), 1e-12)
            candidate_effective_count_trace[t] = 1.0 / np.maximum(np.sum(last_choice_probs**2, axis=1), 1e-12)
            candidate_policy_score_std_trace[t] = np.std(last_choice_probs, axis=-1)

        batch_traj_states.append(np.stack(traj_states, axis=0) if traj_states else np.zeros((0, aug_state_dim), dtype=np.float64))
        batch_traj_choices.append(np.stack(traj_choices, axis=0) if traj_choices else np.zeros((0, action_dim), dtype=np.int64))
        batch_traj_masks.append(np.stack(traj_masks, axis=0) if traj_masks else np.zeros((0, action_dim), dtype=np.float64))
        batch_branch_is_learned.append(1.0 if slot_from_learned else 0.0)
        batch_adv.append(advantage)
        batch_indices.append(t)
        if len(batch_adv) >= rl_cfg.batch_size:
            flush_batch()

    flush_batch()
    print()

    slnr_proxy_trace = np.stack([
        compute_pmi_sinr_proxy_from_precoder(
            pmi_feedback["q_vectors"][tt], slnr_baseline["precoders"][tt], noise_power, eps=rl_cfg.reward_sinr_eps
        )
        for tt in range(cfg.num_slots)
    ], axis=0)

    return {
        "throughput": throughput,
        "reference_like_rate_baseline": reference_like_rate_baseline_trace,
        "reference_like_precoder_distance": reference_like_precoder_distance_trace,
        "reference_like_baseline_update_weight": reference_like_baseline_update_weight_trace,
        "reward": reward_trace,
        "rate_delta": rate_delta_trace,
        "rate_delta_baseline": rate_delta_baseline_trace,
        "rate_delta_slnr": rate_delta_baseline_trace if rl_cfg.reference_precoder == "slnr" else throughput - slnr_baseline["throughput"],
        "proxy_sinr": proxy_sinr_trace,
        "zf_proxy_sinr": zf_proxy_sinr_trace,
        "proxy_sinr_ratio": proxy_sinr_ratio_trace,
        "actual_sinr_ratio": actual_sinr_ratio_trace,
        "proxy_sinr_ratio_baseline": proxy_sinr_ratio_baseline_trace,
        "proxy_sinr_ratio_slnr": proxy_sinr_ratio_baseline_trace if rl_cfg.reference_precoder == "slnr" else proxy_sinr_trace / np.maximum(slnr_proxy_trace, rl_cfg.reward_sinr_eps),
        "actual_sinr_ratio_baseline": actual_sinr_ratio_baseline_trace,
        "actual_sinr_ratio_slnr": actual_sinr_ratio_baseline_trace if rl_cfg.reference_precoder == "slnr" else (actual_sinr_ratio_trace * zf_baseline["sinr"]) / np.maximum(slnr_baseline["sinr"], rl_cfg.reward_sinr_eps),
        "advantage": advantage_trace,
        "reward_baseline": baseline_trace,
        "kappa": kappa_trace,
        "beat_zf": beat_zf_trace,
        "beat_slnr": beat_slnr_trace,
        "beat_baseline": beat_baseline_trace,
        "oracle_slnr_anchor_rate": np.full(cfg.num_slots, np.nan, dtype=np.float64),
        "oracle_best_pool_rate": np.full(cfg.num_slots, np.nan, dtype=np.float64),
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
        "pmi_span_selected_theta": np.zeros((cfg.num_slots, k), dtype=np.float64),
        "pmi_span_selected_phi": np.zeros((cfg.num_slots, k), dtype=np.float64),
        "pmi_center_theta": np.zeros((cfg.num_slots, k), dtype=np.float64),
        "pmi_center_phi": np.zeros((cfg.num_slots, k), dtype=np.float64),
        "pmi_mean_theta": np.zeros((cfg.num_slots, k), dtype=np.float64),
        "pmi_mean_phi": np.zeros((cfg.num_slots, k), dtype=np.float64),
        "selected_alpha_level": selected_alpha_level_trace,
        "mean_alpha_level": mean_alpha_level_trace,
        "alpha_prob": alpha_prob_trace,
        "selected_direction": selected_direction_trace,
        "direction_prob": direction_prob_trace,
        "stop_prob": stop_prob_trace,
        "selected_stop": selected_stop_trace,
        "num_inner_steps": num_inner_steps_trace,
        "direction_names": np.array(direction_names, dtype=object),
        "alpha_levels": alpha_levels_np,
        "policy_variant": "multi_iteration",
        "wesn_states": wesn_states,
    }

def moving_average(trace: np.ndarray, window_len: int) -> np.ndarray:
    if window_len <= 1 or window_len > trace.size:
        return trace.copy()
    kernel = np.ones(window_len, dtype=np.float64) / window_len
    return np.convolve(trace, kernel, mode="valid")


def reference_throughput_from_delta(results: dict[str, np.ndarray]) -> np.ndarray:
    """Recover the selected reference baseline throughput from stored delta traces."""
    if "reference_throughput" in results:
        return np.asarray(results["reference_throughput"], dtype=np.float64)
    return np.asarray(results["throughput"], dtype=np.float64) - np.asarray(results["rate_delta_baseline"], dtype=np.float64)


def rate_gain_percent_vs_reference(results: dict[str, np.ndarray], eps: float = 1e-12) -> np.ndarray:
    """Return 100 * (R_method - R_ref) / |R_ref| for each slot.

    This is a per-slot percentage trace. For plots, prefer
    smoothed_rate_gain_percent_vs_reference() so the plotted percentage is
    computed from smoothed throughput traces rather than from the mean of
    per-slot ratios.
    """
    delta = np.asarray(results["rate_delta_baseline"], dtype=np.float64)
    ref = reference_throughput_from_delta(results)
    return 100.0 * delta / np.maximum(np.abs(ref), eps)


def smoothed_rate_gain_percent_vs_reference(
    results: dict[str, np.ndarray],
    window_len: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return 100 * (MA{R_method} - MA{R_ref}) / |MA{R_ref}|.

    This avoids averaging per-slot ratios, which can overweight slots where the
    reference baseline is small.
    """
    method_avg = moving_average(np.asarray(results["throughput"], dtype=np.float64), window_len)
    ref_avg = moving_average(reference_throughput_from_delta(results), window_len)
    x_size = min(method_avg.size, ref_avg.size)
    method_avg = method_avg[:x_size]
    ref_avg = ref_avg[:x_size]
    return 100.0 * (method_avg - ref_avg) / np.maximum(np.abs(ref_avg), eps)



def make_action_coord_labels(num_coords: int) -> list[str]:
    """Labels for arbitrary-K per-UE alpha/action coordinates."""
    return [f"UE {i}" for i in range(int(num_coords))]


def subplot_grid(num_items: int) -> tuple[int, int]:
    """Return a compact subplot grid for a dynamic number of items."""
    num_items = max(1, int(num_items))
    ncols = min(3, num_items)
    nrows = int(np.ceil(num_items / ncols))
    return nrows, ncols


def safe_action_name(index: int) -> str:
    return f"ue{int(index)}"


def save_plots(
    zf_throughput: np.ndarray,
    slnr_throughput: np.ndarray,
    random_vmf_throughput: np.ndarray,
    rl_results: dict[str, np.ndarray],
    output_dir: Path,
    window_len: int,
    wmmse_throughput: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rl_throughput = rl_results["throughput"]
    reward = rl_results["reward"]
    beat_zf = rl_results["beat_zf"]
    beat_baseline = rl_results.get("beat_baseline", None)
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
    selected_alpha_level = rl_results.get("selected_alpha_level", None)
    mean_alpha_level = rl_results.get("mean_alpha_level", None)
    alpha_prob = rl_results.get("alpha_prob", None)
    alpha_levels = rl_results.get("alpha_levels", None)

    zf_avg = moving_average(zf_throughput, window_len)
    slnr_avg = moving_average(slnr_throughput, window_len)
    random_vmf_avg = moving_average(random_vmf_throughput, window_len)
    wmmse_avg = moving_average(wmmse_throughput, window_len) if wmmse_throughput is not None else None
    rl_avg = moving_average(rl_throughput, window_len)
    reward_avg = moving_average(reward, window_len)
    beat_zf_avg = moving_average(beat_zf, window_len)
    beat_baseline_avg = moving_average(beat_baseline, window_len) if beat_baseline is not None else None
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
    ax1.plot(x_tput, slnr_avg, lw=1.5, label="SLNR baseline")
    if wmmse_avg is not None:
        ax1.plot(x_tput[:wmmse_avg.size], wmmse_avg, lw=1.5, label="WMMSE full-CSI ceiling")
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
    ax2.set_title("WESN RL Reward Across Time")
    ax2.set_xlabel("Slot index")
    ax2.set_ylabel("Reward vs selected reference")
    ax2.grid(True, alpha=0.35)
    fig2.tight_layout()
    fig2.savefig(output_dir / "esn_vmf_rl_reward_across_time.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.plot(np.arange(1, beat_zf_avg.size + 1), beat_zf_avg, lw=1.5)
    ax3.set_title("Fraction of Slots Where WESN Beats ZF")
    ax3.set_xlabel("Slot index")
    ax3.set_ylabel("Moving-average fraction")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.35)
    fig3.tight_layout()
    fig3.savefig(output_dir / "esn_vmf_rl_fraction_beats_zf.png", dpi=150)
    plt.close(fig3)

    if beat_baseline_avg is not None:
        fig3b, ax3b = plt.subplots(figsize=(8, 4.5))
        ax3b.plot(np.arange(1, beat_baseline_avg.size + 1), beat_baseline_avg, lw=1.5)
        ax3b.set_title("Fraction of Slots Where WESN Beats Reference Baseline")
        ax3b.set_xlabel("Slot index")
        ax3b.set_ylabel("Moving-average fraction")
        ax3b.set_ylim(0.0, 1.0)
        ax3b.grid(True, alpha=0.35)
        fig3b.tight_layout()
        fig3b.savefig(output_dir / "esn_vmf_rl_fraction_beats_baseline.png", dpi=150)
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
        ax12.plot(np.arange(1, learned_policy_probability_avg.size + 1), learned_policy_probability_avg, lw=1.5, label="Learned perturbation branch")
        ax12.plot(np.arange(1, dk_policy_probability_avg.size + 1), dk_policy_probability_avg, lw=1.5, label="Exact reference-precoder branch")
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
        ax13.set_title("Fraction of Executed Slots from Learned PMI-Span Branch")
        ax13.set_xlabel("Slot index")
        ax13.set_ylabel("Moving-average fraction")
        ax13.set_ylim(0.0, 1.0)
        ax13.grid(True, alpha=0.35)
        fig13.tight_layout()
        fig13.savefig(output_dir / "hybrid_mixture_executed_learned_branch.png", dpi=150)
        plt.close(fig13)

    pmi_theta = rl_results.get("pmi_span_selected_theta", None)
    pmi_phi = rl_results.get("pmi_span_selected_phi", None)
    if pmi_theta is not None and pmi_phi is not None:
        fig14, ax14 = plt.subplots(figsize=(8, 4.5))
        for ku in range(pmi_theta.shape[1]):
            theta_avg = moving_average(pmi_theta[:, ku], window_len)
            ax14.plot(np.arange(1, theta_avg.size + 1), theta_avg, lw=1.5, label=f"UE {ku}")
        ax14.set_title("Selected PMI-Span Theta Across Time")
        ax14.set_xlabel("Slot index")
        ax14.set_ylabel(r"$	heta_k$")
        ax14.grid(True, alpha=0.35)
        ax14.legend(loc="best")
        fig14.tight_layout()
        fig14.savefig(output_dir / "pmi_span_selected_theta_across_time.png", dpi=150)
        plt.close(fig14)

        fig15, ax15 = plt.subplots(figsize=(8, 4.5))
        for ku in range(pmi_phi.shape[1]):
            phi_avg = moving_average(pmi_phi[:, ku], window_len)
            ax15.plot(np.arange(1, phi_avg.size + 1), phi_avg, lw=1.5, label=f"UE {ku}")
        ax15.set_title("Selected PMI-Span Phi Across Time")
        ax15.set_xlabel("Slot index")
        ax15.set_ylabel(r"$\phi_k$")
        ax15.grid(True, alpha=0.35)
        ax15.legend(loc="best")
        fig15.tight_layout()
        fig15.savefig(output_dir / "pmi_span_selected_phi_across_time.png", dpi=150)
        plt.close(fig15)


    # ------------------------------------------------------------------
    # Discrete-alpha diagnostics.
    # 1) Histogram: how often each coordinate selected each alpha level.
    # 2) Probability heatmaps: how the WESN categorical distribution over
    #    alpha levels evolves over time for each coordinate.
    # ------------------------------------------------------------------
    if selected_alpha_level is not None and alpha_levels is not None:
        alpha_levels_arr = np.asarray(alpha_levels, dtype=np.float64)
        selected_alpha_arr = np.asarray(selected_alpha_level, dtype=np.float64)
        coord_labels = make_action_coord_labels(selected_alpha_arr.shape[1])
        nrows, ncols = subplot_grid(selected_alpha_arr.shape[1])

        fig_alpha_hist, axes_alpha_hist = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), sharey=True)
        axes_alpha_hist = np.atleast_1d(axes_alpha_hist).reshape(-1)
        for coord_idx in range(selected_alpha_arr.shape[1]):
            ax = axes_alpha_hist[coord_idx]
            counts = np.array([
                np.mean(np.isclose(selected_alpha_arr[:, coord_idx], level))
                for level in alpha_levels_arr
            ], dtype=np.float64)
            ax.bar(np.arange(alpha_levels_arr.size), counts)
            ax.set_title(f"Selected alpha levels: {coord_labels[coord_idx]}")
            ax.set_xticks(np.arange(alpha_levels_arr.size))
            ax.set_xticklabels([f"{a:g}" for a in alpha_levels_arr], rotation=45)
            ax.set_ylabel("Selection fraction")
            ax.grid(True, alpha=0.3)
        for ax in axes_alpha_hist[selected_alpha_arr.shape[1]:]:
            ax.axis("off")
        fig_alpha_hist.tight_layout()
        fig_alpha_hist.savefig(output_dir / "alpha_level_selection_histograms.png", dpi=150)
        plt.close(fig_alpha_hist)

    if alpha_prob is not None and alpha_levels is not None:
        alpha_levels_arr = np.asarray(alpha_levels, dtype=np.float64)
        alpha_prob_arr = np.asarray(alpha_prob, dtype=np.float64)
        coord_labels = make_action_coord_labels(alpha_prob_arr.shape[1])

        def moving_average_2d(trace_2d: np.ndarray, win: int) -> np.ndarray:
            if win <= 1 or win > trace_2d.shape[0]:
                return trace_2d.copy()
            kernel = np.ones(win, dtype=np.float64) / win
            return np.stack([
                np.convolve(trace_2d[:, m], kernel, mode="valid")
                for m in range(trace_2d.shape[1])
            ], axis=1)

        heatmap_window = max(1, int(window_len))
        for coord_idx in range(alpha_prob_arr.shape[1]):
            prob_smooth = moving_average_2d(alpha_prob_arr[:, coord_idx, :], heatmap_window)
            fig_alpha_prob, ax_alpha_prob = plt.subplots(figsize=(9, 4.8))
            im = ax_alpha_prob.imshow(
                prob_smooth.T,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            ax_alpha_prob.set_title(f"Alpha-level probability heatmap: {coord_labels[coord_idx]}")
            ax_alpha_prob.set_xlabel("Slot index")
            ax_alpha_prob.set_ylabel("Alpha level")
            ax_alpha_prob.set_yticks(np.arange(alpha_levels_arr.size))
            ax_alpha_prob.set_yticklabels([f"{a:g}" for a in alpha_levels_arr])
            fig_alpha_prob.colorbar(im, ax=ax_alpha_prob, label="Probability")
            fig_alpha_prob.tight_layout()
            safe_name = safe_action_name(coord_idx)
            fig_alpha_prob.savefig(output_dir / f"alpha_probability_heatmap_{safe_name}.png", dpi=150)
            plt.close(fig_alpha_prob)


def save_two_policy_comparison_plots(
    zf_throughput: np.ndarray,
    slnr_throughput: np.ndarray,
    single_results: dict[str, np.ndarray],
    multi_results: dict[str, np.ndarray],
    output_dir: Path,
    window_len: int,
    wmmse_throughput: np.ndarray | None = None,
) -> None:
    """Save comparison plots for the one-direction and multi-direction policies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zf_avg = moving_average(zf_throughput, window_len)
    slnr_avg = moving_average(slnr_throughput, window_len)
    wmmse_avg = moving_average(wmmse_throughput, window_len) if wmmse_throughput is not None else None
    single_tput_avg = moving_average(single_results["throughput"], window_len)
    multi_tput_avg = moving_average(multi_results["throughput"], window_len)
    x_size = min(zf_avg.size, slnr_avg.size, single_tput_avg.size, multi_tput_avg.size)
    if wmmse_avg is not None:
        x_size = min(x_size, wmmse_avg.size)
    x = np.arange(1, x_size + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, zf_avg[:x.size], lw=1.5, label="ZF baseline")
    ax.plot(x, slnr_avg[:x.size], lw=1.5, label="SLNR baseline")
    if wmmse_avg is not None:
        ax.plot(x, wmmse_avg[:x.size], lw=1.5, label="WMMSE full-CSI ceiling")
    ax.plot(x, single_tput_avg[:x.size], lw=1.5, label="WESN: 1 direction")
    ax.plot(x, multi_tput_avg[:x.size], lw=1.5, label="WESN: 3 directions")
    ax.set_title("Throughput Across Time")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Sum-rate [bits/s/Hz]")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_across_time_zf_slnr_single_vs_multi_direction.png", dpi=150)
    plt.close(fig)

    single_reward_avg = moving_average(single_results["reward"], window_len)
    multi_reward_avg = moving_average(multi_results["reward"], window_len)
    xr = np.arange(1, min(single_reward_avg.size, multi_reward_avg.size) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xr, single_reward_avg[:xr.size], lw=1.5, label="WESN: 1 direction")
    ax.plot(xr, multi_reward_avg[:xr.size], lw=1.5, label="WESN: 3 directions")
    ax.set_title("RL Reward Across Time")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Reward vs selected reference")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "reward_across_time_single_vs_multi_direction.png", dpi=150)
    plt.close(fig)

    single_gain_pct_avg = smoothed_rate_gain_percent_vs_reference(single_results, window_len)
    multi_gain_pct_avg = smoothed_rate_gain_percent_vs_reference(multi_results, window_len)
    xg = np.arange(1, min(single_gain_pct_avg.size, multi_gain_pct_avg.size) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xg, single_gain_pct_avg[:xg.size], lw=1.5, label="WESN: 1 direction")
    ax.plot(xg, multi_gain_pct_avg[:xg.size], lw=1.5, label="WESN: 3 directions")
    ax.axhline(0.0, lw=1.0)
    ax.set_title("Throughput Gain vs Reference Baseline Across Time")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Rate gain [%]")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "rate_delta_vs_baseline_single_vs_multi_direction.png", dpi=150)
    plt.close(fig)

    # Mean selected alpha level per UE/action coordinate, comparing both policies.
    num_coords = single_results["selected_alpha_level"].shape[1]
    coord_labels = make_action_coord_labels(num_coords)
    nrows, ncols = subplot_grid(num_coords)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)
    for coord_idx in range(num_coords):
        ax = axes[coord_idx]
        ax.plot(
            moving_average(single_results["selected_alpha_level"][:, coord_idx], window_len),
            lw=1.2,
            label="1 direction",
        )
        ax.plot(
            moving_average(multi_results["selected_alpha_level"][:, coord_idx], window_len),
            lw=1.2,
            label="3 directions",
        )
        ax.set_title(f"Selected alpha level: {coord_labels[coord_idx]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    for ax in axes[num_coords:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_alpha_levels_single_vs_multi_direction.png", dpi=150)
    plt.close(fig)

    # Direction-choice diagnostics for the multi-direction policy.
    direction_names = [str(x) for x in multi_results.get("direction_names", np.array(["proxy-rate", "leakage-reducing", "signal-increasing"], dtype=object))]
    selected_direction = multi_results.get("selected_direction", None)
    direction_prob = multi_results.get("direction_prob", None)
    if selected_direction is not None and direction_prob is not None and len(direction_names) > 1:
        nrows, ncols = subplot_grid(selected_direction.shape[1])
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), sharey=True)
        axes = np.atleast_1d(axes).reshape(-1)
        for coord_idx in range(selected_direction.shape[1]):
            ax = axes[coord_idx]
            counts = np.array([np.mean(selected_direction[:, coord_idx] == d) for d in range(len(direction_names))])
            ax.bar(np.arange(len(direction_names)), counts)
            ax.set_title(f"Selected direction: {coord_labels[coord_idx]}")
            ax.set_xticks(np.arange(len(direction_names)))
            ax.set_xticklabels(direction_names, rotation=30, ha="right")
            ax.set_ylabel("Selection fraction")
            ax.grid(True, alpha=0.3)
        for ax in axes[selected_direction.shape[1]:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "multi_direction_selection_histograms.png", dpi=150)
        plt.close(fig)

        for coord_idx in range(direction_prob.shape[1]):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for d, name in enumerate(direction_names):
                ax.plot(moving_average(direction_prob[:, coord_idx, d], window_len), lw=1.4, label=name)
            ax.set_title(f"Direction probabilities across time: {coord_labels[coord_idx]}")
            ax.set_xlabel("Slot index")
            ax.set_ylabel("Probability")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.35)
            ax.legend(loc="best")
            fig.tight_layout()
            safe_name = safe_action_name(coord_idx)
            fig.savefig(output_dir / f"multi_direction_probabilities_{safe_name}.png", dpi=150)
            plt.close(fig)



def save_three_policy_comparison_plots(
    zf_throughput: np.ndarray,
    slnr_throughput: np.ndarray,
    single_results: dict[str, np.ndarray],
    two_results: dict[str, np.ndarray],
    three_results: dict[str, np.ndarray],
    output_dir: Path,
    window_len: int,
    wmmse_throughput: np.ndarray | None = None,
) -> None:
    """Save comparison plots for one-, two-, and three-direction learned policies."""
    output_dir.mkdir(parents=True, exist_ok=True)

    zf_avg = moving_average(zf_throughput, window_len)
    slnr_avg = moving_average(slnr_throughput, window_len)
    wmmse_avg = moving_average(wmmse_throughput, window_len) if wmmse_throughput is not None else None
    single_tput_avg = moving_average(single_results["throughput"], window_len)
    two_tput_avg = moving_average(two_results["throughput"], window_len)
    three_tput_avg = moving_average(three_results["throughput"], window_len)
    throughput_plot_specs = [
        (
            "throughput_across_time_zf_slnr_1dir.png",
            [(single_tput_avg, "WESN: 1 direction")],
        ),
        (
            "throughput_across_time_zf_slnr_2dir.png",
            [(two_tput_avg, "WESN: 2 directions")],
        ),
        (
            "throughput_across_time_zf_slnr_3dir.png",
            [(three_tput_avg, "WESN: 3 directions")],
        ),
        (
            "throughput_across_time_zf_slnr_1dir_2dir.png",
            [
                (single_tput_avg, "WESN: 1 direction"),
                (two_tput_avg, "WESN: 2 directions"),
            ],
        ),
        (
            "throughput_across_time_zf_slnr_1dir_2dir_3dir.png",
            [
                (single_tput_avg, "WESN: 1 direction"),
                (two_tput_avg, "WESN: 2 directions"),
                (three_tput_avg, "WESN: 3 directions"),
            ],
        ),
    ]
    for filename, wesn_curves in throughput_plot_specs:
        x_size = min(zf_avg.size, slnr_avg.size, *(curve.size for curve, _ in wesn_curves))
        if wmmse_avg is not None:
            x_size = min(x_size, wmmse_avg.size)
        x = np.arange(1, x_size + 1)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, zf_avg[:x_size], lw=1.5, label="ZF baseline")
        ax.plot(x, slnr_avg[:x_size], lw=1.5, label="SLNR baseline")
        if wmmse_avg is not None:
            ax.plot(x, wmmse_avg[:x_size], lw=1.5, label="WMMSE full-CSI ceiling")
        for curve, label in wesn_curves:
            ax.plot(x, curve[:x_size], lw=1.5, label=label)
        ax.set_title("Throughput Across Time")
        ax.set_xlabel("Slot index")
        ax.set_ylabel("Sum-rate [bits/s/Hz]")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    single_reward_avg = moving_average(single_results["reward"], window_len)
    two_reward_avg = moving_average(two_results["reward"], window_len)
    three_reward_avg = moving_average(three_results["reward"], window_len)
    single_delta_avg = smoothed_rate_gain_percent_vs_reference(single_results, window_len)
    two_delta_avg = smoothed_rate_gain_percent_vs_reference(two_results, window_len)
    three_delta_avg = smoothed_rate_gain_percent_vs_reference(three_results, window_len)

    policy_plot_specs = [
        ("1dir", [(single_results, "1 direction")]),
        ("2dir", [(two_results, "2 directions")]),
        ("3dir", [(three_results, "3 directions")]),
        ("1dir_2dir", [(single_results, "1 direction"), (two_results, "2 directions")]),
        (
            "1dir_2dir_3dir",
            [
                (single_results, "1 direction"),
                (two_results, "2 directions"),
                (three_results, "3 directions"),
            ],
        ),
    ]
    reward_avgs = {
        id(single_results): single_reward_avg,
        id(two_results): two_reward_avg,
        id(three_results): three_reward_avg,
    }
    delta_avgs = {
        id(single_results): single_delta_avg,
        id(two_results): two_delta_avg,
        id(three_results): three_delta_avg,
    }

    for slug, policies in policy_plot_specs:
        reward_curves = [(reward_avgs[id(results)], label) for results, label in policies]
        x_size = min(curve.size for curve, _ in reward_curves)
        x = np.arange(1, x_size + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for curve, label in reward_curves:
            ax.plot(x, curve[:x_size], lw=1.5, label=f"WESN: {label}")
        ax.set_title("RL Reward Across Time")
        ax.set_xlabel("Slot index")
        ax.set_ylabel("Reward vs selected reference")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"reward_across_time_{slug}.png", dpi=150)
        plt.close(fig)

        delta_curves = [(delta_avgs[id(results)], label) for results, label in policies]
        x_size = min(curve.size for curve, _ in delta_curves)
        x = np.arange(1, x_size + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for curve, label in delta_curves:
            ax.plot(x, curve[:x_size], lw=1.5, label=f"WESN: {label}")
        ax.axhline(0.0, lw=1.0)
        ax.set_title("Throughput Gain vs Reference Baseline Across Time")
        ax.set_xlabel("Slot index")
        ax.set_ylabel("Rate gain [%]")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"rate_delta_vs_baseline_{slug}.png", dpi=150)
        plt.close(fig)

    num_coords = single_results["selected_alpha_level"].shape[1]
    coord_labels = make_action_coord_labels(num_coords)
    nrows, ncols = subplot_grid(num_coords)
    for slug, policies in policy_plot_specs:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), sharex=True)
        axes = np.atleast_1d(axes).reshape(-1)
        for coord_idx in range(num_coords):
            ax = axes[coord_idx]
            for results, label in policies:
                alpha_avg = moving_average(results["selected_alpha_level"][:, coord_idx], window_len)
                ax.plot(alpha_avg, lw=1.2, label=label)
            ax.set_title(f"Selected alpha level: {coord_labels[coord_idx]}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
        for ax in axes[num_coords:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"selected_alpha_levels_{slug}.png", dpi=150)
        plt.close(fig)

    def save_direction_diagnostics(results: dict[str, np.ndarray], label_slug: str, label_title: str) -> None:
        direction_names = [str(x) for x in results.get("direction_names", np.array([], dtype=object))]
        selected_direction = results.get("selected_direction", None)
        direction_prob = results.get("direction_prob", None)
        if selected_direction is None or direction_prob is None or len(direction_names) <= 1:
            return
        nrows, ncols = subplot_grid(selected_direction.shape[1])
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), sharey=True)
        axes = np.atleast_1d(axes).reshape(-1)
        for coord_idx in range(selected_direction.shape[1]):
            ax = axes[coord_idx]
            counts = np.array([np.mean(selected_direction[:, coord_idx] == d) for d in range(len(direction_names))])
            ax.bar(np.arange(len(direction_names)), counts)
            ax.set_title(f"Selected direction ({label_title}): {coord_labels[coord_idx]}")
            ax.set_xticks(np.arange(len(direction_names)))
            ax.set_xticklabels(direction_names, rotation=30, ha="right")
            ax.set_ylabel("Selection fraction")
            ax.grid(True, alpha=0.3)
        for ax in axes[selected_direction.shape[1]:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"{label_slug}_direction_selection_histograms.png", dpi=150)
        plt.close(fig)

        for coord_idx in range(direction_prob.shape[1]):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for d, name in enumerate(direction_names):
                ax.plot(moving_average(direction_prob[:, coord_idx, d], window_len), lw=1.4, label=name)
            ax.set_title(f"Direction probabilities ({label_title}): {coord_labels[coord_idx]}")
            ax.set_xlabel("Slot index")
            ax.set_ylabel("Probability")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.35)
            ax.legend(loc="best")
            fig.tight_layout()
            safe_name = safe_action_name(coord_idx)
            fig.savefig(output_dir / f"{label_slug}_direction_probabilities_{safe_name}.png", dpi=150)
            plt.close(fig)

    save_direction_diagnostics(two_results, "two_direction", "2 directions")
    save_direction_diagnostics(three_results, "three_direction", "3 directions")

def save_four_policy_comparison_plots(
    zf_throughput: np.ndarray,
    slnr_throughput: np.ndarray,
    single_results: dict[str, np.ndarray],
    two_results: dict[str, np.ndarray],
    three_results: dict[str, np.ndarray],
    multi_iter_results: dict[str, np.ndarray],
    output_dir: Path,
    window_len: int,
    wmmse_throughput: np.ndarray | None = None,
) -> None:
    """Save summary comparison plots including the sequential multi-iteration policy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zf_avg = moving_average(zf_throughput, window_len)
    slnr_avg = moving_average(slnr_throughput, window_len)
    wmmse_avg = moving_average(wmmse_throughput, window_len) if wmmse_throughput is not None else None
    policies = [
        (single_results, "WESN: 1 direction"),
        (two_results, "WESN: 2 directions"),
        (three_results, "WESN: 3 directions"),
        (multi_iter_results, "WESN: multi-iteration"),
    ]

    tput_curves = [(moving_average(res["throughput"], window_len), label) for res, label in policies]
    x_size = min(zf_avg.size, slnr_avg.size, *(curve.size for curve, _ in tput_curves))
    if wmmse_avg is not None:
        x_size = min(x_size, wmmse_avg.size)
    x = np.arange(1, x_size + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, zf_avg[:x_size], lw=1.5, label="ZF baseline")
    ax.plot(x, slnr_avg[:x_size], lw=1.5, label="SLNR baseline")
    if wmmse_avg is not None:
        ax.plot(x, wmmse_avg[:x_size], lw=1.5, label="WMMSE full-CSI ceiling")
    for curve, label in tput_curves:
        ax.plot(x, curve[:x_size], lw=1.5, label=label)
    ax.set_title("Throughput Across Time")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Sum-rate [bits/s/Hz]")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_across_time_zf_slnr_1dir_2dir_3dir_multi_iter.png", dpi=150)
    plt.close(fig)

    reward_curves = [(moving_average(res["reward"], window_len), label) for res, label in policies]
    x_size = min(curve.size for curve, _ in reward_curves)
    x = np.arange(1, x_size + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for curve, label in reward_curves:
        ax.plot(x, curve[:x_size], lw=1.5, label=label)
    ax.set_title("RL Reward Across Time")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Reward vs selected reference")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "reward_across_time_1dir_2dir_3dir_multi_iter.png", dpi=150)
    plt.close(fig)

    delta_curves = [(moving_average(rate_gain_percent_vs_reference(res), window_len), label) for res, label in policies]
    x_size = min(curve.size for curve, _ in delta_curves)
    x = np.arange(1, x_size + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for curve, label in delta_curves:
        ax.plot(x, curve[:x_size], lw=1.5, label=label)
    ax.axhline(0.0, lw=1.0)
    ax.set_title("Throughput Gain vs Reference Baseline Across Time")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Rate gain [%]")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "rate_delta_vs_baseline_1dir_2dir_3dir_multi_iter.png", dpi=150)
    plt.close(fig)

    if "num_inner_steps" in multi_iter_results:
        steps_avg = moving_average(multi_iter_results["num_inner_steps"], window_len)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(np.arange(1, steps_avg.size + 1), steps_avg, lw=1.5)
        ax.set_title("Multi-Iteration Policy: Number of Inner Refinement Steps")
        ax.set_xlabel("Slot index")
        ax.set_ylabel("Moving-average inner steps")
        ax.set_ylim(0.0, max(1.0, np.nanmax(steps_avg) * 1.1))
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_dir / "multi_iteration_num_inner_steps.png", dpi=150)
        plt.close(fig)

    if "stop_prob" in multi_iter_results:
        stop_prob = np.asarray(multi_iter_results["stop_prob"], dtype=np.float64)
        coord_labels = make_action_coord_labels(stop_prob.shape[1])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for ku in range(stop_prob.shape[1]):
            ax.plot(moving_average(stop_prob[:, ku], window_len), lw=1.4, label=coord_labels[ku])
        ax.set_title("Multi-Iteration Policy: Stop Probability")
        ax.set_xlabel("Slot index")
        ax.set_ylabel("Probability")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "multi_iteration_stop_probability.png", dpi=150)
        plt.close(fig)

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
        "channel_model": cfg.channel_model,
        "carrier_frequency_hz": cfg.carrier_frequency_hz,
        "subcarrier_spacing_hz": cfg.subcarrier_spacing_hz,
        "slot_duration_s": cfg.slot_duration_s,
        "ue_speed_kmh": cfg.ue_speed_kmh,
        "bs_height_m": cfg.bs_height_m,
        "ue_height_m": cfg.ue_height_m,
        "cell_radius_m": cfg.cell_radius_m,
        "min_ue_distance_m": cfg.min_ue_distance_m,
        "sionna_scenario": cfg.sionna_scenario,
        "sionna_o2i_model": cfg.sionna_o2i_model,
        "sionna_enable_pathloss": cfg.sionna_enable_pathloss,
        "sionna_enable_shadow_fading": cfg.sionna_enable_shadow_fading,
        "seed": cfg.seed,
        "total_tx_power": cfg.total_tx_power,
        "rate_receiver_model": "local_effective_lmmse_hkp_v1",
        "pmi_feedback_mode": cfg.pmi_feedback_mode,
        "type_ii_feedback_architecture": cfg.type_ii_feedback_architecture,
        "type_ii_nfft": cfg.type_ii_nfft,
        "type_ii_num_ofdm_symbols": cfg.type_ii_num_ofdm_symbols,
    }


def _cache_value_to_slug(value: object) -> str:
    """Convert a cache-setting value into a filesystem-safe filename fragment."""
    if isinstance(value, float):
        value_str = f"{value:.8g}"
    else:
        value_str = str(value)
    value_str = value_str.replace("-", "m").replace(".", "p")
    safe_chars = []
    for ch in value_str:
        if ch.isalnum() or ch in {"_", "="}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_")


def channel_pmi_cache_stem(cfg: SimConfig) -> str:
    """Filename stem encoding all settings that affect generated channels or PMI."""
    parts = [
        f"nt{cfg.num_tx_antennas}",
        f"ue{cfg.num_users}",
        f"nrx{cfg.num_rx_antennas_per_user}",
        f"streams{cfg.streams_per_user}",
        f"slots{cfg.num_slots}",
        f"snr{_cache_value_to_slug(cfg.snr_db)}",
        f"ch{_cache_value_to_slug(cfg.channel_model)}",
        f"rho{_cache_value_to_slug(cfg.temporal_correlation)}",
        f"fc{_cache_value_to_slug(cfg.carrier_frequency_hz)}",
        f"vel{_cache_value_to_slug(cfg.ue_speed_kmh)}",
        f"cell{_cache_value_to_slug(cfg.cell_radius_m)}",
        f"seed{cfg.seed}",
        f"ptx{_cache_value_to_slug(cfg.total_tx_power)}",
        f"pmi{_cache_value_to_slug(cfg.pmi_feedback_mode)}",
    ]
    if cfg.pmi_feedback_mode == "type_ii":
        parts.extend(
            [
                f"arch{_cache_value_to_slug(cfg.type_ii_feedback_architecture)}",
                f"nfft{cfg.type_ii_nfft}",
                f"nsym{cfg.type_ii_num_ofdm_symbols}",
            ]
        )
    return "channel_pmi_" + "_".join(parts)


def baseline_cache_stem(cfg: SimConfig, baseline_name: str, extra_parts: tuple[str, ...] = ()) -> str:
    """Filename stem for channel-dependent baseline caches.

    The channel/PMI stem already encodes simulation dimensions, num_slots, seed,
    SNR, channel model, PMI mode, and other settings that change deterministic
    baseline traces.  Prefixing it with the baseline name lets multiple
    conventional baselines and slot counts coexist in the same cache directory.
    """
    parts = [baseline_name, channel_pmi_cache_stem(cfg)]
    parts.extend(extra_parts)
    return "_".join(parts)


def load_or_generate_channels_and_pmi(
    cfg: SimConfig,
    cache_dir: Path,
    force_recompute: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load cached channels/PMI when the parameterized cache file matches cfg.

    The cache filename itself contains the channel-generation and PMI settings,
    so changing num_users, num_slots, SNR, rho, seed, PMI mode, Type-II OFDM
    dimensions, etc. naturally points to a different cache file.  The sidecar
    JSON metadata is an additional integrity check.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = channel_pmi_cache_stem(cfg)
    data_path = cache_dir / f"{stem}.npz"
    meta_path = cache_dir / f"{stem}_meta.json"
    metadata = sim_cache_metadata(cfg) | {"cache_kind": "channels_and_pmi", "cache_stem": stem}

    can_load = (
        not force_recompute
        and data_path.exists()
        and _metadata_matches(meta_path, metadata)
    )
    if can_load:
        print(f"Loading cached channels and PMI from {data_path}")
        with np.load(data_path) as data:
            channels = data["channels"]
            pmi_feedback = {
                "features": data["pmi_features"],
                "q_vectors": data["pmi_q_vectors"],
            }
        expected_channel_shape = (
            cfg.num_slots,
            cfg.num_users,
            cfg.num_rx_antennas_per_user,
            cfg.num_tx_antennas,
        )
        expected_q_shape = (cfg.num_slots, cfg.num_users, cfg.num_tx_antennas)
        if channels.shape != expected_channel_shape:
            raise ValueError(
                f"Cached channel shape {channels.shape} does not match expected {expected_channel_shape}. "
                "Delete the cache file or rerun with --force-recompute-channel-pmi."
            )
        if pmi_feedback["q_vectors"].shape != expected_q_shape:
            raise ValueError(
                f"Cached PMI q-vector shape {pmi_feedback['q_vectors'].shape} does not match expected {expected_q_shape}. "
                "Delete the cache file or rerun with --force-recompute-channel-pmi."
            )
        return channels, pmi_feedback

    if force_recompute:
        print("Forcing channel/PMI cache regeneration.")
    else:
        print("Cached channels and PMI not found or incompatible; generating once.")
    rng_channels = np.random.default_rng(cfg.seed)
    channels = simulate_channels(cfg, rng_channels)
    pmi_feedback = build_pmi_feedback_trace(cfg, channels)
    np.savez_compressed(
        data_path,
        channels=channels,
        pmi_features=pmi_feedback["features"],
        pmi_q_vectors=pmi_feedback["q_vectors"],
    )
    _write_metadata(meta_path, metadata)
    print(f"Saved channels and PMI cache to {data_path}")
    return channels, pmi_feedback


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
    pmi_feedback: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Load ZF baseline from cache if compatible; otherwise compute and cache it.

    This avoids recomputing the deterministic ZF baseline when repeated runs use
    the same simulation configuration and channel seed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = baseline_cache_stem(cfg, "zf")
    meta_path = cache_dir / f"{stem}_meta.json"
    metadata = sim_cache_metadata(cfg)

    paths = {
        "throughput": cache_dir / f"{stem}_throughput_trace.npy",
        "sinr": cache_dir / f"{stem}_sinr_trace.npy",
        "precoders": cache_dir / f"{stem}_precoders_trace.npy",
        "pmi_features": cache_dir / f"{stem}_pmi_features_trace.npy",
    }

    can_load = (
        not force_recompute
        and _metadata_matches(meta_path, metadata)
        and all(path.exists() for path in paths.values())
    )
    if can_load:
        print(f"Loading cached ZF baseline from {cache_dir / stem}")
        return {name: np.load(path) for name, path in paths.items()}

    print("Cached ZF baseline not found or incompatible; recomputing.")
    zf_results = run_zf_baseline(cfg, channels, pmi_feedback=pmi_feedback)
    for name, path in paths.items():
        np.save(path, zf_results[name])
    _write_metadata(meta_path, metadata)
    return zf_results



def load_or_run_slnr_baseline(
    cfg: SimConfig,
    channels: np.ndarray,
    cache_dir: Path,
    force_recompute: bool = False,
    pmi_feedback: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Load PMI-only SLNR baseline from cache if compatible; otherwise compute it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = baseline_cache_stem(cfg, "slnr")
    meta_path = cache_dir / f"{stem}_meta.json"
    metadata = sim_cache_metadata(cfg) | {"baseline": "pmi_only_slnr"}

    paths = {
        "throughput": cache_dir / f"{stem}_throughput_trace.npy",
        "sinr": cache_dir / f"{stem}_sinr_trace.npy",
        "precoders": cache_dir / f"{stem}_precoders_trace.npy",
    }

    can_load = (
        not force_recompute
        and _metadata_matches(meta_path, metadata)
        and all(path.exists() for path in paths.values())
    )
    if can_load:
        print(f"Loading cached SLNR baseline from {cache_dir / stem}")
        return {name: np.load(path) for name, path in paths.items()}

    print("Cached SLNR baseline not found or incompatible; recomputing.")
    slnr_results = run_slnr_baseline(cfg, channels, pmi_feedback=pmi_feedback)
    for name, path in paths.items():
        np.save(path, slnr_results[name])
    _write_metadata(meta_path, metadata)
    return slnr_results


def load_or_run_wmmse_baseline(
    cfg: SimConfig,
    channels: np.ndarray,
    cache_dir: Path,
    force_recompute: bool = False,
    max_iters: int = 50,
    tol: float = 1e-5,
) -> dict[str, np.ndarray]:
    """Load perfect-full-CSI WMMSE baseline from cache if compatible; otherwise compute it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = baseline_cache_stem(
        cfg,
        "wmmse_full_csi",
        extra_parts=(
            f"iters{int(max_iters)}",
            f"tol{_cache_value_to_slug(float(tol))}",
        ),
    )
    meta_path = cache_dir / f"{stem}_meta.json"
    metadata = sim_cache_metadata(cfg) | {
        "baseline": "wmmse_full_csi",
        "wmmse_max_iters": int(max_iters),
        "wmmse_tol": float(tol),
        "rate_receiver": "mmse_sinr_maximizing",
    }

    paths = {
        "throughput": cache_dir / f"{stem}_throughput_trace.npy",
        "sinr": cache_dir / f"{stem}_sinr_trace.npy",
        "precoders": cache_dir / f"{stem}_precoders_trace.npy",
    }

    can_load = (
        not force_recompute
        and _metadata_matches(meta_path, metadata)
        and all(path.exists() for path in paths.values())
    )
    if can_load:
        print(f"Loading cached full-CSI WMMSE baseline from {cache_dir / stem}")
        return {name: np.load(path) for name, path in paths.items()}

    print("Cached full-CSI WMMSE baseline not found or incompatible; recomputing.")
    wmmse_results = run_wmmse_baseline(cfg, channels, max_iters=max_iters, tol=tol)
    for name, path in paths.items():
        np.save(path, wmmse_results[name])
    _write_metadata(meta_path, metadata)
    return wmmse_results


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



# Central selector for which top-level precoding strategies are evaluated,
# saved, printed, and included in the selected-algorithm throughput plot.
# RL algorithms still need the selected reference baselines (ZF/SLNR) as
# internal dependencies; those dependencies may be computed even when omitted
# from this list, but they are not treated as requested top-level algorithms.
AVAILABLE_ALGORITHMS: tuple[str, ...] = (
    "zf",
    "slnr",
    "wmmse",
    # "random_vmf",
    # "wesn_single_direction",
    "wesn_two_direction",
    # "wesn_multi_direction",
    "wesn_multi_iteration",
)
DEFAULT_ALGORITHMS: tuple[str, ...] = AVAILABLE_ALGORITHMS
ALGORITHM_LABELS: dict[str, str] = {
    "zf": "ZF baseline",
    "slnr": "SLNR baseline",
    "wmmse": "WMMSE full-CSI ceiling",
    "random_vmf": "Random vMF baseline",
    "wesn_single_direction": "WESN: 1 direction",
    "wesn_two_direction": "WESN: 2 directions",
    "wesn_multi_direction": "WESN: 3 directions",
    "wesn_multi_iteration": "WESN: multi-iteration",
}
ALGORITHM_OUTPUT_PREFIXES: dict[str, str] = {
    "zf": "zf",
    "slnr": "slnr",
    "wmmse": "wmmse_full_csi",
    "random_vmf": "random_vmf_baseline",
    "wesn_single_direction": "single_direction",
    "wesn_two_direction": "two_direction",
    "wesn_multi_direction": "multi_direction",
    "wesn_multi_iteration": "multi_iteration",
}
RL_ALGORITHMS: frozenset[str] = frozenset(
    {
        "wesn_single_direction",
        "wesn_two_direction",
        "wesn_multi_direction",
        "wesn_multi_iteration",
    }
)


def parse_algorithm_selection(raw_algorithms: str | None) -> tuple[str, ...]:
    """Parse a comma-separated algorithm list.

    Use "all" (the default) for all algorithms in DEFAULT_ALGORITHMS.
    """
    if raw_algorithms is None or raw_algorithms.strip() == "" or raw_algorithms.strip().lower() == "all":
        return DEFAULT_ALGORITHMS
    selected = tuple(item.strip().lower() for item in raw_algorithms.split(",") if item.strip())
    unknown = sorted(set(selected) - set(AVAILABLE_ALGORITHMS))
    if unknown:
        raise ValueError(
            f"Unknown algorithm(s): {unknown}. Valid choices are: {', '.join(AVAILABLE_ALGORITHMS)}"
        )
    # Preserve user order while removing duplicates.
    return tuple(dict.fromkeys(selected))


def save_selected_algorithm_throughput_plot(
    algorithm_results: dict[str, dict[str, np.ndarray]],
    algorithms: tuple[str, ...],
    output_dir: Path,
    window_len: int,
) -> None:
    """Plot only the algorithms requested by the algorithms selector."""
    throughput_curves: list[tuple[str, np.ndarray]] = []
    for name in algorithms:
        results = algorithm_results.get(name)
        if results is None or "throughput" not in results:
            continue
        label = ALGORITHM_LABELS.get(name, name)
        throughput_curves.append((label, moving_average(results["throughput"], window_len)))
        if name == "wesn_multi_iteration" and "reference_like_rate_baseline" in results:
            throughput_curves.append((f"CQI-based baseline", moving_average(results["reference_like_rate_baseline"], window_len)))

    if not throughput_curves:
        return

    x_size = min(curve.size for _, curve in throughput_curves)
    x = np.arange(1, x_size + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, curve in throughput_curves:
        ax.plot(x, curve[:x_size], lw=1.5, label=label)
    ax.set_title("Throughput Across Time")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Sum-rate [bits/s/Hz]")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_across_time_selected_algorithms.png", dpi=150)
    plt.close(fig)


def save_selected_algorithm_rate_delta_plot(
    algorithm_results: dict[str, dict[str, np.ndarray]],
    algorithms: tuple[str, ...],
    output_dir: Path,
    window_len: int,
) -> None:
    """Plot rate-gain percentage vs each selected RL algorithm's reference baseline.

    Only algorithms with a rate_delta_baseline trace are included. Conventional
    baselines such as ZF/SLNR/WMMSE usually do not have a meaningful
    per-algorithm reference delta, so they are skipped here.
    """
    delta_curves: list[tuple[str, np.ndarray]] = []
    for name in algorithms:
        results = algorithm_results.get(name)
        if results is None or "rate_delta_baseline" not in results or "throughput" not in results:
            continue
        delta_curves.append((ALGORITHM_LABELS.get(name, name), smoothed_rate_gain_percent_vs_reference(results, window_len)))

    if not delta_curves:
        return

    x_size = min(curve.size for _, curve in delta_curves)
    x = np.arange(1, x_size + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, curve in delta_curves:
        ax.plot(x, curve[:x_size], lw=1.5, label=label)
    ax.axhline(0.0, lw=1.0)
    ax.set_title("Throughput Gain vs Reference Baseline Across Time")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Rate gain [%]")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "rate_delta_vs_baseline_selected_algorithms.png", dpi=150)
    plt.close(fig)


def save_selected_algorithm_reward_plot(
    algorithm_results: dict[str, dict[str, np.ndarray]],
    algorithms: tuple[str, ...],
    output_dir: Path,
    window_len: int,
) -> None:
    """Plot reward for selected RL algorithms only."""
    reward_curves: list[tuple[str, np.ndarray]] = []
    for name in algorithms:
        results = algorithm_results.get(name)
        if results is None or "reward" not in results:
            continue
        reward_curves.append((ALGORITHM_LABELS.get(name, name), moving_average(results["reward"], window_len)))

    if not reward_curves:
        return

    x_size = min(curve.size for _, curve in reward_curves)
    x = np.arange(1, x_size + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, curve in reward_curves:
        ax.plot(x, curve[:x_size], lw=1.5, label=label)
    ax.set_title("RL Reward Across Time")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Reward vs selected reference")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "reward_across_time_selected_algorithms.png", dpi=150)
    plt.close(fig)


def clean_existing_plots(output_dir: Path) -> None:
    """Remove stale plot files so old algorithm curves do not appear after changing algorithms.

    This only removes PNG files under output_dir. Cached channels/baselines and saved
    NumPy traces are intentionally preserved.
    """
    if not output_dir.exists():
        return
    for png_path in output_dir.rglob("*.png"):
        try:
            png_path.unlink()
        except OSError:
            pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple MU-MIMO ZF/SLNR baselines + batched WESN reference-precoder perturbation RL")
    parser.add_argument("--num-slots", type=int, default=10000)
    parser.add_argument("--window-len", type=int, default=500)
    parser.add_argument("--num-users", type=int, default=4, help="Number of scheduled UEs/users.")
    parser.add_argument("--multi-iter-max-steps", type=int, default=20, help="Maximum number of inner refinement iterations for the sequential multi-iteration WESN policy.")
    parser.add_argument(
        "--reference-precoder",
        type=str,
        default="slnr",
        choices=["zf", "slnr"],
        help=(
            "Reference precoder used both as the perturbation center and as the "
            "baseline inside the RL reward. Use slnr to perturb/reward relative "
            "to SLNR, or zf to perturb/reward relative to ZF."
        ),
    )
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--num-tx-antennas", type=int, default=4, help="Number of transmit antennas at the BS/virtual array.")
    parser.add_argument("--num-rx-antennas-per-user", type=int, default=2, help="Number of receive antennas per UE.")
    parser.add_argument(
        "--channel-model",
        type=str,
        default="sionna_uma",
        choices=["toy_ar1", "sionna_umi", "sionna_uma", "sionna_rma"],
        help="Channel generator. toy_ar1 is the original i.i.d. AR(1) toy model; sionna_* uses Sionna TR 38.901 channels.",
    )
    parser.add_argument("--carrier-frequency-hz", type=float, default=3.5e9, help="Carrier frequency used by the Sionna channel model.")
    parser.add_argument("--subcarrier-spacing-hz", type=float, default=30e3, help="Subcarrier spacing used when converting Sionna CIR to an OFDM channel.")
    parser.add_argument("--slot-duration-s", type=float, default=1e-3, help="Time step between RL slots for Sionna Doppler/time evolution.")
    parser.add_argument("--ue-speed-kmh", type=float, default=10.0, help="UE speed used by the Sionna topology.")
    parser.add_argument("--bs-height-m", type=float, default=25.0, help="BS height for Sionna topology.")
    parser.add_argument("--ue-height-m", type=float, default=1.5, help="UE height for Sionna topology.")
    parser.add_argument("--cell-radius-m", type=float, default=60.0, help="Maximum UE distance from BS for Sionna topology.")
    parser.add_argument("--min-ue-distance-m", type=float, default=50.0, help="Minimum UE distance from BS for Sionna topology.")
    parser.add_argument("--sionna-o2i-model", type=str, default="low", choices=["low", "high"], help="Sionna outdoor-to-indoor model parameter. UEs are currently placed outdoors.")
    parser.add_argument("--sionna-enable-pathloss", action="store_true", help="Enable Sionna pathloss. Disabled by default to keep channel-power comparisons close to the toy model.")
    parser.add_argument("--sionna-enable-shadow-fading", action="store_true", help="Enable Sionna shadow fading. Disabled by default to keep channel-power comparisons close to the toy model.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rl_precoder_max_sum_rate"))
    parser.add_argument(
        "--keep-old-plots",
        action="store_true",
        help=(
            "Deprecated/no-op compatibility flag. Existing PNG plots are now kept by default "
            "so image viewers do not close when open plot files are removed."
        ),
    )
    parser.add_argument(
        "--clear-old-plots",
        action="store_true",
        help=(
            "Delete existing PNG plots under output_dir/plots before saving new plots. "
            "Use this only when you explicitly want to remove stale plot files."
        ),
    )
    parser.add_argument(
        "--save-legacy-comparison-plots",
        action="store_true",
        help="Also save the older comparison/diagnostic plot suites. These may include dependency baselines such as ZF/SLNR inside WESN diagnostic subdirectories.",
    )
    parser.add_argument("--baseline-cache-dir", type=Path, default=None)
    parser.add_argument("--channel-pmi-cache-dir", type=Path, default=None, help="Directory used to cache generated channels and PMI traces. Defaults to output_dir/channel_pmi_cache.")
    parser.add_argument("--force-recompute-channel-pmi", action="store_true", help="Ignore cached channels/PMI and regenerate them before running baselines/RL.")
    parser.add_argument("--force-recompute-baselines", action="store_true")
    parser.add_argument("--wmmse-max-iters", type=int, default=50, help="Maximum WMMSE block-coordinate iterations per slot for the full-CSI oracle baseline.")
    parser.add_argument("--wmmse-tol", type=float, default=1e-5, help="Relative sum-rate convergence tolerance for the full-CSI WMMSE oracle baseline.")
    parser.add_argument(
        "--pmi-feedback-mode",
        type=str,
        default="type_ii",
        choices=["right_singular_vectors", "type_ii"],
        help="PMI source: original unquantized right singular vectors or actual Type-II PMI from dmimo.mimo.quantized_CSI_feedback.",
    )

    # WESN knobs. The reservoir update is the same as the vanilla ESN; the
    # readout is augmented with a window of recent PMI inputs as skip connections.
    # W_in and W_res are fixed; only W_out is learned.
    parser.add_argument("--reservoir-size", type=int, default=128)
    parser.add_argument("--spectral-radius", type=float, default=0.8)
    parser.add_argument("--input-scale", type=float, default=0.15)
    parser.add_argument(
        "--skip-window-length",
        type=int,
        default=8,
        help="Number of recent PMI feature vectors included in the WESN skip/readout connection. Use 1 for the one-step skip connection.",
    )

    # RL/training knobs.
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-out", type=float, default=3e-2)
    parser.add_argument("--fixed-kappa", type=float, default=3.0)
    parser.add_argument("--reward-baseline-beta", type=float, default=0.99, help="Deprecated/ignored: no moving-average advantage baseline is used.")
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
        default="cqi_reference_rate_log_ratio",
        choices=[
            "rate_log_ratio",
            "cqi_reference_rate_log_ratio",
        ],
        help=(
            "Scalar reward used for REINFORCE updates. rate_log_ratio uses the "
            "continuous reference rate; cqi_reference_rate_log_ratio quantizes "
            "the reference UE SINRs to CQI levels first."
        ),
    )
    parser.add_argument("--reward-sinr-eps", type=float, default=1e-12)
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
    parser.add_argument("--alpha-slnr-prior-weight", type=float, default=0.0, help="Optional fixed PMI-SLNR prior weight added to WESN candidate logits. Default 0 means WESN learns candidate logits directly.")
    parser.add_argument("--pmi-span-num-theta", type=int, default=8, help="Number of theta grid points for the user-centered PMI-span RL policy. Theta=0 is included once.")
    parser.add_argument("--pmi-span-num-phi", type=int, default=8, help="Number of relative phase grid points for the user-centered PMI-span RL policy.")
    parser.add_argument("--pmi-span-theta-max", type=float, default=float(np.pi / 2.0), help="Maximum theta for the PMI-span policy. Default pi/2 searches the full K=2 user-centered PMI-span family.")
    parser.add_argument("--gaussian-raw-std-theta", type=float, default=0.30, help="Fixed raw-coordinate exploration std for theta coordinates in the 4D diagonal Gaussian policy.")
    parser.add_argument("--gaussian-raw-std-phi", type=float, default=0.30, help="Fixed raw-coordinate exploration std for phi coordinates in the 4D diagonal Gaussian policy.")
    parser.add_argument("--alpha-step-max-theta", type=float, default=0.25, help="Maximum physical theta step used by alpha_max * tanh(u_alpha) along the PMI proxy-gradient direction.")
    parser.add_argument("--alpha-step-max-phi", type=float, default=0.75, help="Maximum physical phi step used by alpha_max * alpha_level along the PMI proxy-gradient direction.")
    parser.add_argument(
        "--num-alpha-levels",
        type=int,
        default=16,
        help="Number of uniformly spaced alpha levels from -1 to 1 for the discrete angle-step policy.",
    )
    parser.add_argument("--alpha-zero-logit-bias", type=float, default=2.0, help="Fixed logit bias added to the zero-alpha level so the initial policy favors the exact reference-precoder-center action.")
    parser.add_argument("--multi-iter-step-penalty", type=float, default=0.0, help="Optional penalty subtracted from the terminal reward per executed multi-iteration refinement step.")
    parser.add_argument("--no-multi-iter-stop-action", action="store_true", help="Disable the stop action in the multi-iteration policy; the policy then always runs for H_max inner steps.")
    parser.add_argument("--gaussian-init-theta", type=float, default=0.40, help="Initial physical theta mean for each UE before conversion to raw sigmoid coordinates.")
    parser.add_argument("--gaussian-init-phi", type=float, default=float(np.pi), help="Initial physical phi mean for each UE before conversion to raw sigmoid coordinates.")
    parser.add_argument("--no-exact-anchor-candidates", action="store_true", help="Disable deterministic exact SLNR / SLNR-residual / PMI candidates in the candidate pool.")
    parser.add_argument("--joint-pmi-aux-loss-weight", type=float, default=0.0, help="Weight of the joint PMI-only full-precoder auxiliary ranking loss. Use 0 to disable.")
    parser.add_argument("--joint-pmi-aux-temperature", type=float, default=0.5, help="Softmax temperature used to convert joint PMI proxy rates into auxiliary target weights.")
    parser.add_argument("--num-joint-pmi-aux-candidates", type=int, default=32, help="Number of full-precoder candidate combinations sampled per slot for the joint PMI auxiliary loss.")
    parser.add_argument("--enable-oracle-pool-diagnostics", action="store_true", help="Expensively score all M^K full-precoder combinations from the candidate pools and log the best actual rate.")
    parser.add_argument("--initial-learned-policy-probability", type=float, default=1.0, help="Initial reuse probability for the learned reference-precoder perturbation branch. The exact reference-precoder branch starts with probability 1-p.")
    parser.add_argument("--lr-mixture", type=float, default=0, help="Learning rate for the trainable mixture/reuse probability logit.")
    parser.add_argument("--positive-rate-bonus-lambda", type=float, default=0.0, help="Scale lambda for the positive rate-delta bonus: lambda * max(rate - reference_rate, 0)^p.")
    parser.add_argument("--positive-rate-bonus-power", type=float, default=0.5, help="Exponent p for the positive rate-delta bonus. Default 0.5 amplifies small positive deltas.")

    parser.add_argument(
        "--algorithms",
        type=str,
        default="all",
        help=(
            "Comma-separated top-level algorithms to evaluate/plot. Use 'all' for the default full set. "
            "Valid names: " + ", ".join(AVAILABLE_ALGORITHMS)
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(
        num_tx_antennas=args.num_tx_antennas,
        num_users=args.num_users,
        num_rx_antennas_per_user=args.num_rx_antennas_per_user,
        num_slots=args.num_slots,
        snr_db=args.snr_db,
        temporal_correlation=args.rho,
        channel_model=args.channel_model,
        carrier_frequency_hz=args.carrier_frequency_hz,
        subcarrier_spacing_hz=args.subcarrier_spacing_hz,
        slot_duration_s=args.slot_duration_s,
        ue_speed_kmh=args.ue_speed_kmh,
        bs_height_m=args.bs_height_m,
        ue_height_m=args.ue_height_m,
        cell_radius_m=args.cell_radius_m,
        min_ue_distance_m=args.min_ue_distance_m,
        sionna_scenario=args.channel_model.split("sionna_", 1)[1] if args.channel_model.startswith("sionna_") else "umi",
        sionna_o2i_model=args.sionna_o2i_model,
        sionna_enable_pathloss=args.sionna_enable_pathloss,
        sionna_enable_shadow_fading=args.sionna_enable_shadow_fading,
        seed=args.seed,
        pmi_feedback_mode=args.pmi_feedback_mode,
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
        reference_precoder=args.reference_precoder,
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
        pmi_span_num_theta=args.pmi_span_num_theta,
        pmi_span_num_phi=args.pmi_span_num_phi,
        pmi_span_theta_max=args.pmi_span_theta_max,
        gaussian_raw_std_theta=args.gaussian_raw_std_theta,
        gaussian_raw_std_phi=args.gaussian_raw_std_phi,
        alpha_step_max_theta=args.alpha_step_max_theta,
        alpha_step_max_phi=args.alpha_step_max_phi,
        alpha_level_grid=tuple(np.linspace(-1.0, 1.0, args.num_alpha_levels)),
        alpha_zero_logit_bias=args.alpha_zero_logit_bias,
        multi_iter_max_steps=args.multi_iter_max_steps,
        multi_iter_step_penalty=args.multi_iter_step_penalty,
        multi_iter_include_stop_action=not args.no_multi_iter_stop_action,
        gaussian_init_theta=args.gaussian_init_theta,
        gaussian_init_phi=args.gaussian_init_phi,
        joint_pmi_aux_loss_weight=args.joint_pmi_aux_loss_weight,
        enable_oracle_pool_diagnostics=args.enable_oracle_pool_diagnostics,
        joint_pmi_aux_temperature=args.joint_pmi_aux_temperature,
        num_joint_pmi_aux_candidates=args.num_joint_pmi_aux_candidates,
        initial_learned_policy_probability=args.initial_learned_policy_probability,
        lr_mixture=args.lr_mixture,
        positive_rate_bonus_lambda=args.positive_rate_bonus_lambda,
        positive_rate_bonus_power=args.positive_rate_bonus_power,
    )

    # Use separate RNG streams so loading cached channels/PMI or baselines does
    # not change the stochastic trajectory of the RL policy. This keeps repeated
    # runs comparable.
    rng_random_vmf = np.random.default_rng(cfg.seed + 10_000)
    rng_rl = np.random.default_rng(cfg.seed + 20_000)
    torch.manual_seed(cfg.seed)

    baseline_cache_dir = args.baseline_cache_dir
    if baseline_cache_dir is None:
        baseline_cache_dir = args.output_dir / "baseline_cache"

    channel_pmi_cache_dir = args.channel_pmi_cache_dir
    if channel_pmi_cache_dir is None:
        channel_pmi_cache_dir = args.output_dir / "channel_pmi_cache"

    channels, pmi_feedback = load_or_generate_channels_and_pmi(
        cfg=cfg,
        cache_dir=channel_pmi_cache_dir,
        force_recompute=args.force_recompute_channel_pmi,
    )

    algorithms = parse_algorithm_selection(args.algorithms)
    algorithm_set = set(algorithms)
    requested_rl = algorithm_set & RL_ALGORITHMS

    # ZF/SLNR are internal dependencies for all WESN policies because the WESN
    # state uses PMI features from ZF and the policy/reward can use ZF or SLNR
    # as the reference. They are only saved/plotted/printed as top-level
    # algorithms if they are included in `algorithms`.
    need_zf = "zf" in algorithm_set or bool(requested_rl)
    need_slnr = "slnr" in algorithm_set or bool(requested_rl)
    if requested_rl and rl_cfg.reference_precoder == "slnr":
        need_slnr = True
    if requested_rl and rl_cfg.reference_precoder == "zf":
        need_zf = True

    algorithm_results: dict[str, dict[str, np.ndarray]] = {}
    zf_results: dict[str, np.ndarray] | None = None
    slnr_results: dict[str, np.ndarray] | None = None

    if need_zf:
        zf_results = load_or_run_zf_baseline(
            cfg=cfg,
            channels=channels,
            cache_dir=baseline_cache_dir,
            force_recompute=args.force_recompute_baselines,
            pmi_feedback=pmi_feedback,
        )
        if "zf" in algorithm_set:
            algorithm_results["zf"] = zf_results

    if need_slnr:
        slnr_results = load_or_run_slnr_baseline(
            cfg=cfg,
            channels=channels,
            cache_dir=baseline_cache_dir,
            force_recompute=args.force_recompute_baselines,
            pmi_feedback=pmi_feedback,
        )
        if "slnr" in algorithm_set:
            algorithm_results["slnr"] = slnr_results

    if "wmmse" in algorithm_set:
        algorithm_results["wmmse"] = load_or_run_wmmse_baseline(
            cfg=cfg,
            channels=channels,
            cache_dir=baseline_cache_dir,
            force_recompute=args.force_recompute_baselines,
            max_iters=args.wmmse_max_iters,
            tol=args.wmmse_tol,
        )

    if "random_vmf" in algorithm_set:
        algorithm_results["random_vmf"] = load_or_run_random_vmf_baseline(
            cfg=cfg,
            channels=channels,
            rng=rng_random_vmf,
            fixed_kappa=rl_cfg.fixed_kappa,
            cache_dir=baseline_cache_dir,
            force_recompute=args.force_recompute_baselines,
        )

    if requested_rl:
        if zf_results is None or slnr_results is None:
            raise RuntimeError("WESN algorithms require ZF and SLNR dependency results.")

    if "wesn_single_direction" in algorithm_set:
        algorithm_results["wesn_single_direction"] = run_wesn_policy_rl(
            cfg,
            rl_cfg,
            channels,
            zf_results,
            slnr_results,
            rng_rl,
            policy_variant="single_direction",
            pmi_feedback=pmi_feedback,
        )

    # Use separate RNG streams so each learned policy has an independent but
    # reproducible exploration trajectory while using the same channels and
    # deterministic baselines.
    if "wesn_two_direction" in algorithm_set:
        rng_rl_two = np.random.default_rng(cfg.seed + 30_000)
        torch.manual_seed(cfg.seed + 1)
        algorithm_results["wesn_two_direction"] = run_wesn_policy_rl(
            cfg,
            rl_cfg,
            channels,
            zf_results,
            slnr_results,
            rng_rl_two,
            policy_variant="two_direction",
            pmi_feedback=pmi_feedback,
        )

    if "wesn_multi_direction" in algorithm_set:
        rng_rl_multi = np.random.default_rng(cfg.seed + 40_000)
        torch.manual_seed(cfg.seed + 2)
        algorithm_results["wesn_multi_direction"] = run_wesn_policy_rl(
            cfg,
            rl_cfg,
            channels,
            zf_results,
            slnr_results,
            rng_rl_multi,
            policy_variant="multi_direction",
            pmi_feedback=pmi_feedback,
        )

    if "wesn_multi_iteration" in algorithm_set:
        rng_rl_multi_iter = np.random.default_rng(cfg.seed + 50_000)
        torch.manual_seed(cfg.seed + 3)
        algorithm_results["wesn_multi_iteration"] = run_wesn_policy_rl_multi_iteration(
            cfg,
            rl_cfg,
            channels,
            zf_results,
            slnr_results,
            rng_rl_multi_iter,
            pmi_feedback=pmi_feedback,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npy_output_dir = args.output_dir / "npy_files"
    plot_output_dir = args.output_dir / "plots"
    npy_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_old_plots:
        clean_existing_plots(plot_output_dir)

    def save_result_arrays(prefix: str, results: dict[str, np.ndarray]) -> None:
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                np.save(npy_output_dir / f"{prefix}_{key}_trace.npy", value)

    for algorithm_name in algorithms:
        results = algorithm_results.get(algorithm_name)
        if results is None:
            continue
        save_result_arrays(ALGORITHM_OUTPUT_PREFIXES.get(algorithm_name, algorithm_name), results)

    save_selected_algorithm_throughput_plot(
        algorithm_results=algorithm_results,
        algorithms=algorithms,
        output_dir=plot_output_dir,
        window_len=args.window_len,
    )
    save_selected_algorithm_rate_delta_plot(
        algorithm_results=algorithm_results,
        algorithms=algorithms,
        output_dir=plot_output_dir,
        window_len=args.window_len,
    )
    save_selected_algorithm_reward_plot(
        algorithm_results=algorithm_results,
        algorithms=algorithms,
        output_dir=plot_output_dir,
        window_len=args.window_len,
    )

    single_direction_results = algorithm_results.get("wesn_single_direction")
    two_direction_results = algorithm_results.get("wesn_two_direction")
    multi_direction_results = algorithm_results.get("wesn_multi_direction")
    multi_iteration_results = algorithm_results.get("wesn_multi_iteration")
    wmmse_results = algorithm_results.get("wmmse")
    random_vmf_results = algorithm_results.get("random_vmf")

    if args.save_legacy_comparison_plots:
        # Keep the richer legacy comparison plots when all required WESN variants
        # are selected. These comparison plots use only selected WESN policies, and
        # include WMMSE only when WMMSE itself was requested.

        if (
            zf_results is not None
            and slnr_results is not None
            and single_direction_results is not None
            and two_direction_results is not None
            and multi_direction_results is not None
        ):
            save_three_policy_comparison_plots(
                zf_throughput=zf_results["throughput"],
                slnr_throughput=slnr_results["throughput"],
                single_results=single_direction_results,
                two_results=two_direction_results,
                three_results=multi_direction_results,
                output_dir=plot_output_dir,
                window_len=args.window_len,
                wmmse_throughput=wmmse_results["throughput"] if wmmse_results is not None else None,
            )

        if (
            zf_results is not None
            and slnr_results is not None
            and single_direction_results is not None
            and two_direction_results is not None
            and multi_direction_results is not None
            and multi_iteration_results is not None
        ):
            save_four_policy_comparison_plots(
                zf_throughput=zf_results["throughput"],
                slnr_throughput=slnr_results["throughput"],
                single_results=single_direction_results,
                two_results=two_direction_results,
                three_results=multi_direction_results,
                multi_iter_results=multi_iteration_results,
                output_dir=plot_output_dir,
                window_len=args.window_len,
                wmmse_throughput=wmmse_results["throughput"] if wmmse_results is not None else None,
            )

        # Keep the old single-policy diagnostic plots for selected WESN policies.
        # random_vmf_throughput is required by the helper but is not currently drawn
        # on the throughput plot, so use zeros when random_vmf is not requested.
        if zf_results is not None and slnr_results is not None:
            random_vmf_throughput = (
                random_vmf_results["throughput"] if random_vmf_results is not None else np.zeros_like(zf_results["throughput"])
            )
            diagnostic_specs = [
                ("wesn_single_direction", "single_direction_diagnostics"),
                ("wesn_two_direction", "two_direction_diagnostics"),
                ("wesn_multi_direction", "multi_direction_diagnostics"),
                ("wesn_multi_iteration", "multi_iteration_diagnostics"),
            ]
            for algorithm_name, subdir in diagnostic_specs:
                results = algorithm_results.get(algorithm_name)
                if results is None:
                    continue
                save_plots(
                    zf_throughput=zf_results["throughput"],
                    slnr_throughput=slnr_results["throughput"],
                    random_vmf_throughput=random_vmf_throughput,
                    rl_results=results,
                    output_dir=plot_output_dir / subdir,
                    window_len=args.window_len,
                    wmmse_throughput=wmmse_results["throughput"] if wmmse_results is not None else None,
                )

    print("Simple WESN-FB PMI reference-precoder perturbation RL run finished.")
    print(f"Algorithms requested              : {', '.join(algorithms)}")
    for algorithm_name in algorithms:
        results = algorithm_results.get(algorithm_name)
        if results is None or "throughput" not in results:
            continue
        print(
            f"{ALGORITHM_LABELS.get(algorithm_name, algorithm_name):34s}: "
            f"{results['throughput'].mean():.4f} bits/s/Hz"
        )

    if requested_rl:
        print(f"Reference precoder                : {rl_cfg.reference_precoder}")
        for algorithm_name in algorithms:
            if algorithm_name not in RL_ALGORITHMS:
                continue
            results = algorithm_results.get(algorithm_name)
            if results is None:
                continue
            label = ALGORITHM_LABELS.get(algorithm_name, algorithm_name)
            print(f"{label:34s} reward              : {results['reward'].mean():.4f}")
            print(f"{label:34s} delta vs baseline   : {results['rate_delta_baseline'].mean():.4f}")
            print(f"{label:34s} beats baseline frac : {results['beat_baseline'].mean():.4f}")

    if multi_iteration_results is not None:
        print(f"WESN multi-iteration mean inner steps: {multi_iteration_results['num_inner_steps'].mean():.4f}")
        print(f"WESN multi-iteration mean stop fraction per UE: {multi_iteration_results['selected_stop'].mean(axis=0)}")
    if two_direction_results is not None:
        print(f"Mean selected direction index for 2-direction policy: {two_direction_results['selected_direction'].mean(axis=0)}")
        print(f"2-direction names                : {list(two_direction_results['direction_names'])}")
    if multi_direction_results is not None:
        print(f"Mean selected direction index for 3-direction policy: {multi_direction_results['selected_direction'].mean(axis=0)}")
        print(f"3-direction names                : {list(multi_direction_results['direction_names'])}")
    if requested_rl:
        print(f"Alpha levels                     : {rl_cfg.alpha_level_grid}")
        print(f"Alpha zero logit bias            : {rl_cfg.alpha_zero_logit_bias:.4f}")

if __name__ == "__main__":
    main()
