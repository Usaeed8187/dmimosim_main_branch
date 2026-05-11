from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
class OracleConfig:
    # Current/random perturbation oracle.
    random_candidates_per_user: int = 32
    random_candidate_kappa: float = 10.0
    include_random_uniform_candidates: bool = False
    include_random_pmi_candidates: bool = False
    random_uniform_fraction: float = 0.10
    random_pmi_fraction: float = 0.15

    # Structured leakage-gradient oracle.
    leakage_alpha_grid: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00)
    include_negative_leakage_steps: bool = False

    # Two-direction structured oracle: leakage-reduction plus desired-signal
    # increase in the tangent space around the SLNR beam.
    signal_beta_grid: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
    include_negative_signal_steps: bool = False

    # Runtime / plotting.
    window_len: int = 200
    output_dir: Path = Path("results/precoder_oracle_random_vs_structured")


def complex_gaussian(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), eps)


def normalize_columns_equal_power(precoder: np.ndarray, total_tx_power: float) -> np.ndarray:
    k = precoder.shape[1]
    p_col = total_tx_power / max(k, 1)
    out = np.zeros_like(precoder)
    for i in range(k):
        col = precoder[:, i]
        nrm = np.linalg.norm(col)
        if nrm < 1e-12:
            col = np.ones_like(col) / np.sqrt(col.size)
            nrm = np.linalg.norm(col)
        out[:, i] = np.sqrt(p_col) * col / nrm
    return out


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


def pmi_features_from_channels(user_channels: np.ndarray) -> list[np.ndarray]:
    vk_list: list[np.ndarray] = []
    for k in range(user_channels.shape[0]):
        _, _, vh = np.linalg.svd(user_channels[k], full_matrices=True)
        vk_list.append(vh.conj().T)
    return vk_list


def pmi_dominant_vectors_from_channels(user_channels: np.ndarray) -> np.ndarray:
    q_list: list[np.ndarray] = []
    for k in range(user_channels.shape[0]):
        _, _, vh = np.linalg.svd(user_channels[k], full_matrices=True)
        q = vh.conj().T[:, 0]
        q_list.append(unit_norm(q))
    return np.stack(q_list, axis=0)


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
    if streams_per_user != 1:
        raise ValueError("This script currently assumes one stream per UE.")

    q_vectors = []
    for vk in full_vk_list:
        q_vectors.append(unit_norm(vk[:, 0]))

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
        try:
            vk_slnr = np.linalg.solve(leakage, qk)
        except np.linalg.LinAlgError:
            vk_slnr = np.linalg.pinv(leakage) @ qk
        directions[:, k] = unit_norm(vk_slnr)

    return normalize_columns_equal_power(directions, total_tx_power)


def compute_slot_sum_rate(
    user_channels: np.ndarray, precoder: np.ndarray, noise_power: float
) -> tuple[float, np.ndarray]:
    num_users = user_channels.shape[0]
    sinr = np.zeros(num_users, dtype=np.float64)

    for k in range(num_users):
        hk = user_channels[k]
        signal_vec = hk @ precoder[:, k]
        uk = signal_vec / max(float(np.linalg.norm(signal_vec)), 1e-12)

        desired = np.abs(np.vdot(uk, hk @ precoder[:, k])) ** 2
        interference = 0.0
        for j in range(num_users):
            if j != k:
                interference += np.abs(np.vdot(uk, hk @ precoder[:, j])) ** 2
        sinr[k] = desired / (interference + noise_power)

    return float(np.sum(np.log2(1.0 + sinr))), sinr


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


def complex_unit_to_real(q: np.ndarray) -> np.ndarray:
    q = unit_norm(q)
    return np.concatenate([np.real(q), np.imag(q)]).astype(np.float64)


def real_to_complex_unit(x: np.ndarray) -> np.ndarray:
    nt = x.size // 2
    return unit_norm(x[:nt] + 1j * x[nt:])


def apply_random_global_phase(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.exp(1j * rng.uniform(0.0, 2.0 * np.pi)) * v


def sample_phase_randomized_vmf_complex(
    center: np.ndarray, kappa: float, rng: np.random.Generator
) -> np.ndarray:
    x = sample_vmf(complex_unit_to_real(center), kappa, rng)
    v = real_to_complex_unit(x)
    return unit_norm(apply_random_global_phase(v, rng))


def sample_uniform_complex_unit(num_tx_antennas: int, rng: np.random.Generator) -> np.ndarray:
    return unit_norm(complex_gaussian((num_tx_antennas,), rng))


def beams_to_precoder(beams: np.ndarray, total_tx_power: float) -> np.ndarray:
    """Convert unit-norm direction columns to equal-power precoder columns."""
    return normalize_columns_equal_power(beams, total_tx_power)


def build_current_random_candidate_pools(
    slnr_precoder: np.ndarray,
    q_vectors: np.ndarray,
    cfg: SimConfig,
    oracle_cfg: OracleConfig,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Current random-perturbation candidate pool.

    For each UE k, candidate 0 is the exact SLNR direction.  The remaining
    candidates are mostly vMF samples around the SLNR direction.  Optional PMI
    and uniform samples can be enabled to mimic the mixed pools used in the RL
    experiments, but this oracle has no WESN/RL component.
    """
    m = max(1, int(oracle_cfg.random_candidates_per_user))
    pools: list[np.ndarray] = []
    for k in range(cfg.num_users):
        v_slnr = unit_norm(slnr_precoder[:, k])
        qk = unit_norm(q_vectors[k])
        cand = np.zeros((m, cfg.num_tx_antennas), dtype=np.complex128)
        cand[0] = v_slnr
        idx = 1

        n_uniform = 0
        n_pmi = 0
        if oracle_cfg.include_random_uniform_candidates:
            n_uniform = int(round(oracle_cfg.random_uniform_fraction * (m - 1)))
        if oracle_cfg.include_random_pmi_candidates:
            n_pmi = int(round(oracle_cfg.random_pmi_fraction * (m - 1)))
        n_uniform = max(0, min(n_uniform, m - idx))
        n_pmi = max(0, min(n_pmi, m - idx - n_uniform))
        n_slnr = m - idx - n_uniform - n_pmi

        for _ in range(n_uniform):
            cand[idx] = sample_uniform_complex_unit(cfg.num_tx_antennas, rng)
            idx += 1
        for _ in range(n_pmi):
            cand[idx] = sample_phase_randomized_vmf_complex(
                qk, oracle_cfg.random_candidate_kappa, rng
            )
            idx += 1
        for _ in range(n_slnr):
            cand[idx] = sample_phase_randomized_vmf_complex(
                v_slnr, oracle_cfg.random_candidate_kappa, rng
            )
            idx += 1
        pools.append(cand)
    return pools


def leakage_gradient_direction(
    v_slnr: np.ndarray,
    q_vectors: np.ndarray,
    user_index: int,
    eps: float = 1e-12,
) -> np.ndarray | None:
    """Return the unit-norm tangent direction that locally reduces PMI leakage.

    Leakage for beam k is L_k(v) = v^H A_k v, where
    A_k = sum_{j != k} q_j q_j^H.  The leakage-increasing gradient direction
    is A_k v.  The tangent component is (I - v v^H) A_k v.  Moving in the
    negative of this tangent component reduces leakage to first order.
    """
    nt = v_slnr.size
    v = unit_norm(v_slnr)
    a_mat = np.zeros((nt, nt), dtype=np.complex128)
    for j in range(q_vectors.shape[0]):
        if j != user_index:
            qj = unit_norm(q_vectors[j])
            a_mat += np.outer(qj, qj.conj())

    grad = a_mat @ v
    tangent_grad = grad - v * np.vdot(v, grad)  # (I - vv^H) grad
    nrm = np.linalg.norm(tangent_grad)
    if nrm < eps:
        return None
    return -tangent_grad / nrm




def desired_signal_gradient_direction(
    v_slnr: np.ndarray,
    q_vectors: np.ndarray,
    user_index: int,
    eps: float = 1e-12,
) -> np.ndarray | None:
    """Return the unit-norm tangent direction that locally increases PMI signal.

    Desired PMI gain for beam k is S_k(v) = v^H B_k v, where
    B_k = q_k q_k^H.  The signal-increasing gradient direction is B_k v.
    The tangent component is (I - v v^H) B_k v.  Moving in the positive
    tangent component increases desired PMI gain to first order.
    """
    v = unit_norm(v_slnr)
    qk = unit_norm(q_vectors[user_index])
    b_mat = np.outer(qk, qk.conj())
    grad = b_mat @ v
    tangent_grad = grad - v * np.vdot(v, grad)
    nrm = np.linalg.norm(tangent_grad)
    if nrm < eps:
        return None
    return tangent_grad / nrm


def _signed_grid(values: tuple[float, ...], include_negative: bool) -> list[float]:
    vals = list(values)
    if include_negative:
        positive = [a for a in vals if a > 0]
        vals = sorted(set([-a for a in positive] + vals))
    if 0.0 not in vals:
        vals = [0.0] + vals
    return list(vals)

def build_structured_leakage_candidate_pools(
    slnr_precoder: np.ndarray,
    q_vectors: np.ndarray,
    cfg: SimConfig,
    oracle_cfg: OracleConfig,
) -> list[np.ndarray]:
    """Structured leakage-gradient candidate pool.

    For each UE k, candidates are generated as

        v_k(alpha) = normalize(v_k^SLNR + alpha u_k),

    where u_k = -g_perp/||g_perp|| is the leakage-reducing tangent direction.
    """
    alphas = _signed_grid(
        oracle_cfg.leakage_alpha_grid, oracle_cfg.include_negative_leakage_steps
    )

    pools: list[np.ndarray] = []
    for k in range(cfg.num_users):
        v_slnr = unit_norm(slnr_precoder[:, k])
        u = leakage_gradient_direction(v_slnr, q_vectors, k)
        cand = np.zeros((len(alphas), cfg.num_tx_antennas), dtype=np.complex128)
        if u is None:
            # No leakage-gradient direction is available; every candidate is SLNR.
            for i in range(len(alphas)):
                cand[i] = v_slnr
        else:
            for i, alpha in enumerate(alphas):
                cand[i] = unit_norm(v_slnr + float(alpha) * u)
        pools.append(cand)
    return pools



def build_two_direction_candidate_pools(
    slnr_precoder: np.ndarray,
    q_vectors: np.ndarray,
    cfg: SimConfig,
    oracle_cfg: OracleConfig,
) -> list[np.ndarray]:
    """Two-direction structured candidate pool.

    For each UE k, candidates are generated as

        v_k(alpha, beta) = normalize(v_k^SLNR + alpha u_leak,k + beta u_sig,k),

    where u_leak,k is the leakage-reducing tangent direction and u_sig,k is
    the desired-signal-increasing tangent direction.  This creates a local 2D
    structured surface around the SLNR beam instead of a 1D leakage-only curve.
    """
    alphas = _signed_grid(
        oracle_cfg.leakage_alpha_grid, oracle_cfg.include_negative_leakage_steps
    )
    betas = _signed_grid(
        oracle_cfg.signal_beta_grid, oracle_cfg.include_negative_signal_steps
    )

    pools: list[np.ndarray] = []
    for k in range(cfg.num_users):
        v_slnr = unit_norm(slnr_precoder[:, k])
        u_leak = leakage_gradient_direction(v_slnr, q_vectors, k)
        u_sig = desired_signal_gradient_direction(v_slnr, q_vectors, k)

        cand_list: list[np.ndarray] = []
        for alpha in alphas:
            for beta in betas:
                step = np.zeros_like(v_slnr)
                if u_leak is not None:
                    step = step + float(alpha) * u_leak
                if u_sig is not None:
                    step = step + float(beta) * u_sig
                cand_list.append(unit_norm(v_slnr + step))

        # Remove near-duplicates caused by zero gradients or repeated zero pairs.
        unique: list[np.ndarray] = []
        for c in cand_list:
            is_duplicate = False
            for u in unique:
                # Compare phase-invariant beam similarity.
                if np.abs(np.vdot(u, c)) > 1.0 - 1e-10:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(c)
        pools.append(np.stack(unique, axis=0))
    return pools


def oracle_best_rate_from_pools(
    user_channels: np.ndarray,
    candidate_pools: list[np.ndarray],
    total_tx_power: float,
    noise_power: float,
) -> tuple[float, tuple[int, ...], np.ndarray]:
    k = len(candidate_pools)
    best_rate = -np.inf
    best_combo: tuple[int, ...] | None = None
    best_sinr: np.ndarray | None = None

    for combo in product(*[range(pool.shape[0]) for pool in candidate_pools]):
        beams = np.zeros((candidate_pools[0].shape[1], k), dtype=np.complex128)
        for ku, ci in enumerate(combo):
            beams[:, ku] = candidate_pools[ku][ci]
        precoder = beams_to_precoder(beams, total_tx_power)
        rate, sinr = compute_slot_sum_rate(user_channels, precoder, noise_power)
        if rate > best_rate:
            best_rate = rate
            best_combo = tuple(combo)
            best_sinr = sinr

    assert best_combo is not None and best_sinr is not None
    return float(best_rate), best_combo, best_sinr


def moving_average(trace: np.ndarray, window_len: int) -> np.ndarray:
    if window_len <= 1 or window_len > trace.size:
        return trace.copy()
    kernel = np.ones(window_len, dtype=np.float64) / window_len
    return np.convolve(trace, kernel, mode="valid")


def save_plots(results: dict[str, np.ndarray], oracle_cfg: OracleConfig) -> None:
    out = oracle_cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    w = oracle_cfg.window_len

    zf_avg = moving_average(results["zf_rate"], w)
    slnr_avg = moving_average(results["slnr_rate"], w)
    random_avg = moving_average(results["random_oracle_rate"], w)
    structured_avg = moving_average(results["structured_oracle_rate"], w)
    two_direction_avg = moving_average(results["two_direction_oracle_rate"], w)
    x = np.arange(1, zf_avg.size + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, zf_avg, lw=1.4, label="ZF baseline")
    ax.plot(x, slnr_avg, lw=1.4, label="PMI-SLNR baseline")
    ax.plot(x, random_avg, lw=1.4, label="Random-perturbation oracle")
    ax.plot(x, structured_avg, lw=1.4, label="Structured leakage oracle")
    ax.plot(x, two_direction_avg, lw=1.4, label="Two-direction structured oracle")
    ax.set_title("Oracle Rate Across Time")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Sum-rate [bits/s/Hz]")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "oracle_rate_across_time.png", dpi=160)
    plt.close(fig)

    random_gain = results["random_oracle_rate"] - results["slnr_rate"]
    structured_gain = results["structured_oracle_rate"] - results["slnr_rate"]
    two_direction_gain = results["two_direction_oracle_rate"] - results["slnr_rate"]
    random_gain_avg = moving_average(random_gain, w)
    structured_gain_avg = moving_average(structured_gain, w)
    two_direction_gain_avg = moving_average(two_direction_gain, w)
    xg = np.arange(1, random_gain_avg.size + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xg, random_gain_avg, lw=1.4, label="Random oracle - SLNR")
    ax.plot(xg, structured_gain_avg, lw=1.4, label="Structured oracle - SLNR")
    ax.plot(xg, two_direction_gain_avg, lw=1.4, label="Two-direction oracle - SLNR")
    ax.axhline(0.0, lw=1.0, linestyle="--")
    ax.set_title("Oracle Gain Over PMI-SLNR Across Time")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Gain [bits/s/Hz]")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "oracle_gain_over_slnr_across_time.png", dpi=160)
    plt.close(fig)


def run_oracles(cfg: SimConfig, oracle_cfg: OracleConfig) -> dict[str, np.ndarray]:
    rng_channels = np.random.default_rng(cfg.seed)
    rng_oracle = np.random.default_rng(cfg.seed + 20_000)
    channels = simulate_channels(cfg, rng_channels)
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))

    zf_rate = np.zeros(cfg.num_slots, dtype=np.float64)
    slnr_rate = np.zeros(cfg.num_slots, dtype=np.float64)
    random_oracle_rate = np.zeros(cfg.num_slots, dtype=np.float64)
    structured_oracle_rate = np.zeros(cfg.num_slots, dtype=np.float64)
    two_direction_oracle_rate = np.zeros(cfg.num_slots, dtype=np.float64)
    random_best_combo = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.int64)
    structured_best_combo = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.int64)
    two_direction_best_combo = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.int64)

    for t in range(cfg.num_slots):
        print(f"Oracle slot {t + 1} / {cfg.num_slots}", end="\r")
        h_t = channels[t]
        vk_list = pmi_features_from_channels(h_t)
        q_vectors = pmi_dominant_vectors_from_channels(h_t)

        p_zf = build_zf_precoder_from_pmi(vk_list, cfg.streams_per_user, cfg.total_tx_power)
        zf_rate[t], _ = compute_slot_sum_rate(h_t, p_zf, noise_power)

        p_slnr = build_slnr_precoder_from_pmi(
            vk_list, cfg.streams_per_user, cfg.total_tx_power, noise_power
        )
        slnr_rate[t], _ = compute_slot_sum_rate(h_t, p_slnr, noise_power)

        random_pools = build_current_random_candidate_pools(
            slnr_precoder=p_slnr,
            q_vectors=q_vectors,
            cfg=cfg,
            oracle_cfg=oracle_cfg,
            rng=rng_oracle,
        )
        rate, combo, _ = oracle_best_rate_from_pools(h_t, random_pools, cfg.total_tx_power, noise_power)
        random_oracle_rate[t] = rate
        random_best_combo[t] = np.array(combo, dtype=np.int64)

        structured_pools = build_structured_leakage_candidate_pools(
            slnr_precoder=p_slnr,
            q_vectors=q_vectors,
            cfg=cfg,
            oracle_cfg=oracle_cfg,
        )
        rate, combo, _ = oracle_best_rate_from_pools(h_t, structured_pools, cfg.total_tx_power, noise_power)
        structured_oracle_rate[t] = rate
        structured_best_combo[t] = np.array(combo, dtype=np.int64)

        two_direction_pools = build_two_direction_candidate_pools(
            slnr_precoder=p_slnr,
            q_vectors=q_vectors,
            cfg=cfg,
            oracle_cfg=oracle_cfg,
        )
        rate, combo, _ = oracle_best_rate_from_pools(
            h_t, two_direction_pools, cfg.total_tx_power, noise_power
        )
        two_direction_oracle_rate[t] = rate
        two_direction_best_combo[t] = np.array(combo, dtype=np.int64)

    print()
    return {
        "zf_rate": zf_rate,
        "slnr_rate": slnr_rate,
        "random_oracle_rate": random_oracle_rate,
        "structured_oracle_rate": structured_oracle_rate,
        "two_direction_oracle_rate": two_direction_oracle_rate,
        "random_best_combo": random_best_combo,
        "structured_best_combo": structured_best_combo,
        "two_direction_best_combo": two_direction_best_combo,
    }


def parse_alpha_grid(s: str) -> tuple[float, ...]:
    vals = []
    for item in s.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("alpha grid cannot be empty")
    if 0.0 not in vals:
        vals = [0.0] + vals
    return tuple(vals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare random SLNR-perturbation, leakage-correction, and two-direction structured oracle candidate pools."
    )
    parser.add_argument("--num-slots", type=int, default=100)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-tx-antennas", type=int, default=4)
    parser.add_argument("--num-users", type=int, default=2)
    parser.add_argument("--num-rx-antennas-per-user", type=int, default=2)
    parser.add_argument("--total-tx-power", type=float, default=1.0)

    parser.add_argument("--random-candidates-per-user", type=int, default=128)
    parser.add_argument("--random-candidate-kappa", type=float, default=10.0)
    parser.add_argument("--include-random-uniform-candidates", action="store_true", default=False)
    parser.add_argument("--include-random-pmi-candidates", action="store_true", default=False)
    parser.add_argument("--random-uniform-fraction", type=float, default=0.10)
    parser.add_argument("--random-pmi-fraction", type=float, default=0.15)

    parser.add_argument(
        "--leakage-alpha-grid",
        type=str,
        default="0,0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.80,1.00",
        help="Comma-separated alpha values for v(alpha)=normalize(v_SLNR + alpha*u_leak). 0 is added if missing.",
    )
    parser.add_argument("--include-negative-leakage-steps", action="store_true", default=True)
    parser.add_argument(
        "--signal-beta-grid",
        type=str,
        default="0,0.02,0.05,0.10,0.15,0.20,0.30,0.50",
        help="Comma-separated beta values for v(alpha,beta)=normalize(v_SLNR + alpha*u_leak + beta*u_sig). 0 is added if missing.",
    )
    parser.add_argument("--include-negative-signal-steps", action="store_true", default=False)

    parser.add_argument("--window-len", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/precoder_oracle_random_vs_structured"))
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
        seed=args.seed,
        total_tx_power=args.total_tx_power,
    )
    oracle_cfg = OracleConfig(
        random_candidates_per_user=args.random_candidates_per_user,
        random_candidate_kappa=args.random_candidate_kappa,
        include_random_uniform_candidates=args.include_random_uniform_candidates,
        include_random_pmi_candidates=args.include_random_pmi_candidates,
        random_uniform_fraction=args.random_uniform_fraction,
        random_pmi_fraction=args.random_pmi_fraction,
        leakage_alpha_grid=parse_alpha_grid(args.leakage_alpha_grid),
        include_negative_leakage_steps=args.include_negative_leakage_steps,
        signal_beta_grid=parse_alpha_grid(args.signal_beta_grid),
        include_negative_signal_steps=args.include_negative_signal_steps,
        window_len=args.window_len,
        output_dir=args.output_dir,
    )

    oracle_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (oracle_cfg.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"sim": asdict(cfg), "oracle": {**asdict(oracle_cfg), "output_dir": str(oracle_cfg.output_dir)}}, f, indent=2)

    results = run_oracles(cfg, oracle_cfg)
    for name, arr in results.items():
        np.save(oracle_cfg.output_dir / f"{name}.npy", arr)
    save_plots(results, oracle_cfg)

    slnr_mean = results["slnr_rate"].mean()
    random_mean = results["random_oracle_rate"].mean()
    structured_mean = results["structured_oracle_rate"].mean()
    two_direction_mean = results["two_direction_oracle_rate"].mean()

    print("Oracle comparison run finished.")
    print(f"ZF average throughput                   : {results['zf_rate'].mean():.4f} bits/s/Hz")
    print(f"PMI-SLNR baseline throughput            : {slnr_mean:.4f} bits/s/Hz")
    print(f"Random SLNR-perturbation oracle throughput : {random_mean:.4f} bits/s/Hz")
    print(f"Structured leakage-correction oracle throughput: {structured_mean:.4f} bits/s/Hz")
    print(f"Two-direction structured oracle throughput     : {two_direction_mean:.4f} bits/s/Hz")
    print(f"Random oracle gain over SLNR            : {random_mean - slnr_mean:.4f} bits/s/Hz")
    print(f"Structured oracle gain over SLNR        : {structured_mean - slnr_mean:.4f} bits/s/Hz")
    print(f"Two-direction oracle gain over SLNR     : {two_direction_mean - slnr_mean:.4f} bits/s/Hz")
    print(f"Random oracle beats SLNR fraction       : {np.mean(results['random_oracle_rate'] > results['slnr_rate']):.4f}")
    print(f"Structured oracle beats SLNR fraction   : {np.mean(results['structured_oracle_rate'] > results['slnr_rate']):.4f}")
    print(f"Two-direction oracle beats SLNR fraction: {np.mean(results['two_direction_oracle_rate'] > results['slnr_rate']):.4f}")
    print(f"Saved results and plots to              : {oracle_cfg.output_dir}")
    print(f"Rate plot                               : {oracle_cfg.output_dir / 'oracle_rate_across_time.png'}")
    print(f"Gain plot                               : {oracle_cfg.output_dir / 'oracle_gain_over_slnr_across_time.png'}")


if __name__ == "__main__":
    main()
