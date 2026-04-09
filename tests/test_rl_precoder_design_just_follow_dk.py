"""Simple MU-MIMO ZF + hybrid RL precoder testbench (no dMIMO pipeline dependencies).

Scenario:
- N_t = 4 BS antennas
- K = 2 users
- N_r = 2 antennas per user
- 1 stream per user

What this script does:
1) Simulate temporally correlated channels.
2) Run a pure ZF policy using ideal PMI (full V_k from SVD) and record throughput.
3) Run a hybrid contextual-bandit RL policy inspired by the provided LaTeX derivation:
   - State: PMI feedback features
   - Action: precoder sampled from a complex Gaussian learned policy
   - DK branch: ZF-centered narrow Gaussian
   - Mixture policy weight p = sigmoid(alpha)
   - Hard constraint: total transmit-power (Frobenius) normalization for precoders
   - Soft constraint: leakage penalty (no other penalties)
   - Reward: throughput gain above ZF baseline (minus leakage penalty)
4) Save plots/traces to disk.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SimConfig:
    num_tx_antennas: int = 4
    num_users: int = 2
    num_rx_antennas_per_user: int = 2
    streams_per_user: int = 1
    num_slots: int = 300
    snr_db: float = 10.0
    temporal_correlation: float = 0.95
    seed: int = 7
    total_tx_power: float = 1.0


@dataclass
class RLConfig:
    reservoir_size: int = 64
    spectral_radius: float = 0.8
    input_scale: float = 0.15
    sigma_eps: float = 1e-4
    sigma_dk2: float = 1e-3
    lr_mu: float = 8e-4
    lr_alpha: float = 5e-3
    leakage_penalty_weight: float = 0.10
    batch_size: int = 50
    zf_cov_buffer_size: int = 256
    reward_ema_beta: float = 0.98
    advantage_clip: float = 5.0
    advantage_eps: float = 1e-6
    zf_cov_shrinkage: float = 0.05
    cov_jitter: float = 1e-6


def complex_gaussian(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def split_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.real(x)) + 1j * np.tanh(np.imag(x))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))

def normalize_precoder_frobenius(precoder: np.ndarray, total_tx_power: float) -> np.ndarray:
    fro_norm = np.linalg.norm(precoder, ord="fro")
    if fro_norm < 1e-12:
        return precoder
    return np.sqrt(total_tx_power) * precoder / fro_norm


def build_zf_precoder_from_pmi(full_vk_list: list[np.ndarray], streams_per_user: int, total_tx_power: float) -> np.ndarray:
    """Build ZF precoder from user PMI matrices V_k."""
    selected_rows = []
    for vk in full_vk_list:
        vk_dom = vk[:, :streams_per_user]
        selected_rows.append(vk_dom.conj().T)

    z_matrix = np.vstack(selected_rows)
    precoder = np.linalg.pinv(z_matrix)
    return normalize_precoder_frobenius(precoder, total_tx_power=total_tx_power)


def compute_slot_sum_rate(user_channels: np.ndarray, precoder: np.ndarray, noise_power: float) -> tuple[float, np.ndarray]:
    """Compute per-user SINR and slot sum-rate with one stream/user."""
    num_users = user_channels.shape[0]
    sinr = np.zeros(num_users, dtype=np.float64)

    for k in range(num_users):
        hk = user_channels[k]
        signal_vec = hk @ precoder[:, k]
        signal_norm = np.linalg.norm(signal_vec)
        uk = signal_vec / max(signal_norm, 1e-12)

        desired = np.abs(np.vdot(uk, hk @ precoder[:, k])) ** 2
        interference = 0.0
        for j in range(num_users):
            if j != k:
                interference += np.abs(np.vdot(uk, hk @ precoder[:, j])) ** 2

        sinr[k] = desired / (interference + noise_power)

    return float(np.sum(np.log2(1.0 + sinr))), sinr


def compute_leakage(user_channels: np.ndarray, precoder: np.ndarray) -> float:
    """Inter-user leakage penalty: sum_k sum_{j!=k} ||H_k p_j||^2."""
    num_users = user_channels.shape[0]
    leakage = 0.0
    for k in range(num_users):
        hk = user_channels[k]
        for j in range(num_users):
            if j != k:
                leakage += np.linalg.norm(hk @ precoder[:, j]) ** 2
    return float(leakage)


def pmi_features_from_channels(user_channels: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return PMI feature vector and full V_k list."""
    vk_list: list[np.ndarray] = []
    feature_parts = []
    for k in range(user_channels.shape[0]):
        _, _, vh = np.linalg.svd(user_channels[k], full_matrices=True)
        vk = vh.conj().T
        vk_list.append(vk)
        feature_parts.append(np.real(vk).reshape(-1))
        feature_parts.append(np.imag(vk).reshape(-1))

    feat = np.concatenate(feature_parts, axis=0).astype(np.float64)
    return feat, vk_list


def simulate_channels(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate channels for all slots: [T, K, N_r, N_t]."""
    channels = np.zeros(
        (cfg.num_slots, cfg.num_users, cfg.num_rx_antennas_per_user, cfg.num_tx_antennas),
        dtype=np.complex128,
    )

    h_prev = complex_gaussian((cfg.num_users, cfg.num_rx_antennas_per_user, cfg.num_tx_antennas), rng)
    for t in range(cfg.num_slots):

        innovation = complex_gaussian(h_prev.shape, rng)
        h_curr = cfg.temporal_correlation * h_prev + np.sqrt(1 - cfg.temporal_correlation**2) * innovation
        channels[t] = h_curr
        h_prev = h_curr

    return channels


def run_zf_baseline(cfg: SimConfig, channels: np.ndarray) -> dict[str, np.ndarray]:
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))

    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    sinr_traces = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.float64)
    precoders = np.zeros((cfg.num_slots, cfg.num_tx_antennas, cfg.num_users), dtype=np.complex128)
    pmi_features = []

    for t in range(cfg.num_slots):
        feat, vk_list = pmi_features_from_channels(channels[t])
        pmi_features.append(feat)

        p_zf = build_zf_precoder_from_pmi(vk_list, cfg.streams_per_user, cfg.total_tx_power)
        rate, sinr = compute_slot_sum_rate(channels[t], p_zf, noise_power)

        throughput[t] = rate
        sinr_traces[t] = sinr
        precoders[t] = p_zf

    return {
        "throughput": throughput,
        "sinr": sinr_traces,
        "precoders": precoders,
        "pmi_features": np.stack(pmi_features, axis=0),
    }

def estimate_empirical_complex_cov(samples: np.ndarray, shrinkage: float, jitter: float) -> np.ndarray:
    """Empirical covariance with Ledoit-style shrinkage and diagonal jitter."""
    num_samples, dim = samples.shape
    if num_samples < 2:
        return np.eye(dim, dtype=np.complex128) * (1e-3 + jitter)

    centered = samples - np.mean(samples, axis=0, keepdims=True)
    cov = (centered.conj().T @ centered) / max(num_samples - 1, 1)
    tr = float(np.real(np.trace(cov)))
    target = (tr / max(dim, 1)) * np.eye(dim, dtype=np.complex128)
    cov = (1.0 - shrinkage) * cov + shrinkage * target
    cov = 0.5 * (cov + cov.conj().T)
    cov += jitter * np.eye(dim, dtype=np.complex128)
    return cov


def robust_cholesky(cov: np.ndarray, base_jitter: float) -> np.ndarray:
    """Cholesky with adaptive jitter for numerical stability."""
    dim = cov.shape[0]
    eye = np.eye(dim, dtype=np.complex128)
    jitter = base_jitter
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.conj().T))
    eigvals = np.maximum(np.real(eigvals), base_jitter)
    return eigvecs @ np.diag(np.sqrt(eigvals))


def complex_full_logpdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray, cov_chol: np.ndarray | None = None) -> float:
    """Log-pdf of proper complex Gaussian CN(mu, cov)."""
    dim = x.size
    err = x - mu
    if cov_chol is None:
        cov_chol = robust_cholesky(cov, base_jitter=1e-12)

    y = np.linalg.solve(cov_chol, err)
    quad = float(np.real(np.vdot(y, y)))
    logdet = 2.0 * float(np.sum(np.log(np.abs(np.diag(cov_chol)))))
    return float(-dim * np.log(np.pi) - logdet - quad)


def sample_complex_full_gaussian(mu: np.ndarray, cov_chol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample from CN(mu, cov) given Cholesky factor cov_chol."""
    noise = complex_gaussian(mu.shape, rng)
    return mu + cov_chol @ noise


def complex_diag_logpdf(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    """Log-pdf of proper complex diagonal Gaussian CN(mu, diag(var))."""
    quad = np.sum((np.abs(x - mu) ** 2) / var)
    logdet = np.sum(np.log(var))
    n = x.size
    return float(-n * np.log(np.pi) - logdet - quad)


def sample_complex_gaussian(mu: np.ndarray, var: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = complex_gaussian(mu.shape, rng)
    return mu + np.sqrt(var) * noise


def run_hybrid_rl(
    cfg: SimConfig,
    rl_cfg: RLConfig,
    channels: np.ndarray,
    zf_baseline: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """No-learning policy: sample from DK-derived Gaussian statistics only."""
    if cfg.streams_per_user != 1:
        raise ValueError("This simple RL example supports streams_per_user=1 only.")

    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    num_streams = cfg.num_users * cfg.streams_per_user

    # State/action dimensions.
    action_dim = cfg.num_tx_antennas * num_streams

    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    reward_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    p_trace = np.ones(cfg.num_slots, dtype=np.float64)
    leak_trace = np.zeros(cfg.num_slots, dtype=np.float64)

    zf_buffer = deque(maxlen=max(int(rl_cfg.zf_cov_buffer_size), 2))

    for t in range(cfg.num_slots):

        print("RL Slot {} / {}".format(t + 1, cfg.num_slots), end="\r")

        # Domain-knowledge statistics from the running ZF buffer.
        dk_mu = zf_baseline["precoders"][t].reshape(-1)
        zf_buffer.append(dk_mu.copy())
        dk_cov = estimate_empirical_complex_cov(
            np.asarray(zf_buffer),
            shrinkage=rl_cfg.zf_cov_shrinkage,
            jitter=rl_cfg.cov_jitter,
        )

        # "Learned" policy is set directly from DK statistics (no learning).
        # Mean comes from DK policy; covariance also comes from DK policy.
        mu = dk_mu
        learned_cov = dk_cov + rl_cfg.sigma_eps * np.eye(action_dim, dtype=np.complex128)
        learned_cov_chol = robust_cholesky(learned_cov, base_jitter=rl_cfg.cov_jitter)
        p = 1.0

        # Sample action from DK-statistics Gaussian only.
        action_vec_raw = sample_complex_full_gaussian(mu, learned_cov_chol, rng)

        # Hard constraint: total transmit power normalization.
        action_precoder = action_vec_raw.reshape(cfg.num_tx_antennas, num_streams)
        action_precoder = normalize_precoder_frobenius(action_precoder, cfg.total_tx_power)

        # Environment throughput and leakage.
        rate_rl, _ = compute_slot_sum_rate(channels[t], action_precoder, noise_power)
        leakage = compute_leakage(channels[t], action_precoder)
        reward = (rate_rl - zf_baseline["throughput"][t]) - rl_cfg.leakage_penalty_weight * leakage

        throughput[t] = rate_rl
        reward_trace[t] = reward
        p_trace[t] = p
        leak_trace[t] = leakage

    return {
        "throughput": throughput,
        "reward": reward_trace,
        "p_trace": p_trace,
        "leakage": leak_trace,
    }


def save_results(zf_results: dict[str, np.ndarray], rl_results: dict[str, np.ndarray], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    zf_tput = zf_results["throughput"]
    rl_tput = rl_results["throughput"]

    # Save traces
    zf_trace = output_dir / "zf_throughput_trace.npy"
    rl_trace = output_dir / "dk_stats_only_throughput_trace.npy"
    reward_trace = output_dir / "dk_stats_only_reward_trace.npy"
    learned_policy_prob_trace = output_dir / "dk_stats_only_learned_policy_probability_trace.npy"
    domain_knowledge_prob_trace = output_dir / "dk_stats_only_domain_knowledge_probability_trace.npy"


    np.save(zf_trace, zf_tput)
    np.save(rl_trace, rl_tput)
    np.save(reward_trace, rl_results["reward"])
    np.save(learned_policy_prob_trace, rl_results["p_trace"])
    np.save(domain_knowledge_prob_trace, 1.0 - rl_results["p_trace"])


    w_len = 1000
    zf_tput_avg = np.convolve(zf_tput, np.ones(w_len) / w_len, mode='valid')
    rl_tput_avg = np.convolve(rl_tput, np.ones(w_len) / w_len, mode='valid')
    reward_avg = np.convolve(rl_results["reward"], np.ones(w_len) / w_len, mode='valid')
    learned_policy_prob_avg = np.convolve(rl_results["p_trace"], np.ones(w_len) / w_len, mode='valid')
    domain_knowledge_prob_avg = np.convolve(1.0 - rl_results["p_trace"], np.ones(w_len) / w_len, mode='valid')

    # Throughput plot
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(np.arange(1, len(zf_tput_avg) + 1), zf_tput_avg, lw=1.6, label="ZF baseline")
    ax1.plot(np.arange(1, len(rl_tput_avg) + 1), rl_tput_avg, lw=1.6, label="DK-stats-only policy")
    ax1.set_title("Throughput Across Time")
    ax1.set_xlabel("Slot index")
    ax1.set_ylabel("Sum-rate [bits/s/Hz]")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")
    fig1.tight_layout()

    tput_plot = output_dir / "throughput_across_time_zf_vs_dk_stats_only.png"
    fig1.savefig(tput_plot, dpi=150)
    plt.close(fig1)

    # Reward plot
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(np.arange(1, len(reward_avg) + 1), reward_avg, lw=1.6)
    ax2.set_title("DK-Stats-Only Reward Across Time\n(reward = throughput gain over ZF - leakage penalty)")
    ax2.set_xlabel("Slot index")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.35)
    fig2.tight_layout()

    reward_plot = output_dir / "hybrid_rl_reward_across_time.png"
    fig2.savefig(reward_plot, dpi=150)
    plt.close(fig2)

    # Branch probability plot
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.plot(
        np.arange(1, len(learned_policy_prob_avg) + 1),
        learned_policy_prob_avg,
        lw=1.6,
        label="Learned policy probability",
    )
    ax3.plot(
        np.arange(1, len(domain_knowledge_prob_avg) + 1),
        domain_knowledge_prob_avg,
        lw=1.6,
        label="Domain knowledge probability",
    )
    ax3.set_title("DK-Stats-Only Branch Probabilities Across Time")
    ax3.set_xlabel("Slot index")
    ax3.set_ylabel("Probability")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.35)
    ax3.legend(loc="best")
    fig3.tight_layout()

    probability_plot = output_dir / "dk_stats_only_branch_probabilities_across_time.png"
    fig3.savefig(probability_plot, dpi=150)
    plt.close(fig3)

    return {
        "zf_trace": zf_trace,
        "rl_trace": rl_trace,
        "reward_trace": reward_trace,
        "learned_policy_prob_trace": learned_policy_prob_trace,
        "domain_knowledge_prob_trace": domain_knowledge_prob_trace,
        "throughput_plot": tput_plot,
        "reward_plot": reward_plot,
        "probability_plot": probability_plot,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple MU-MIMO ZF + hybrid RL precoder testbench")
    parser.add_argument("--num-slots", type=int, default=100000, help="Number of simulated time slots")
    parser.add_argument("--snr-db", type=float, default=15.0, help="SNR in dB")
    parser.add_argument("--rho", type=float, default=0.95, help="Temporal channel correlation coefficient")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--reservoir-size", type=int, default=64, help="Fixed ESN reservoir size")
    parser.add_argument("--leakage-weight", type=float, default=0.00, help="Leakage penalty weight in RL reward")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/simple_zf_hybrid_rl_testbench"),
        help="Directory to save traces and plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = SimConfig(
        num_slots=args.num_slots,
        snr_db=args.snr_db,
        temporal_correlation=args.rho,
        seed=args.seed,
    )
    rl_cfg = RLConfig(reservoir_size=args.reservoir_size, leakage_penalty_weight=args.leakage_weight)

    rng = np.random.default_rng(cfg.seed)
    channels = simulate_channels(cfg, rng)

    # 1) Run ZF first to define baseline throughput used in reward.
    zf_results = run_zf_baseline(cfg, channels)

    # 2) Run hybrid RL policy using reward gain above ZF baseline.
    rl_results = run_hybrid_rl(cfg, rl_cfg, channels, zf_results, rng)

    out = save_results(zf_results, rl_results, args.output_dir)

    print("Simple ZF + Hybrid RL testbench finished.")
    print(f"ZF average throughput       : {zf_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"DK Stats Only average throughput: {rl_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"DK Stats Only average reward    : {rl_results['reward'].mean():.4f}")
    print("Saved artifacts:")
    for name, path in out.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()