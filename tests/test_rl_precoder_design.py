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
    lr_sigma: float = 2e-4
    lr_alpha: float = 5e-3
    leakage_penalty_weight: float = 0.10
    batch_size: int = 100
    reward_ema_beta: float = 0.98
    advantage_clip: float = 5.0
    advantage_eps: float = 1e-6

def complex_gaussian(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def split_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.real(x)) + 1j * np.tanh(np.imag(x))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


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
    """Online hybrid policy-gradient training with DK Gaussian branch."""
    if cfg.streams_per_user != 1:
        raise ValueError("This simple RL example supports streams_per_user=1 only.")

    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    num_streams = cfg.num_users * cfg.streams_per_user

    # State/action dimensions.
    pmi_dim = zf_baseline["pmi_features"].shape[1]
    action_dim = cfg.num_tx_antennas * num_streams
    nz = rl_cfg.reservoir_size

    # Fixed ESN reservoir.
    win = rl_cfg.input_scale * complex_gaussian((nz, pmi_dim), rng)
    wres = complex_gaussian((nz, nz), rng)
    eigvals = np.linalg.eigvals(wres)
    wres = (rl_cfg.spectral_radius / max(np.max(np.abs(eigvals)), 1e-12)) * wres

    # Trainable readout parameters.
    w_mu = 0.05 * complex_gaussian((action_dim, nz), rng)
    u_sigma = 0.01 * rng.standard_normal((action_dim, 2 * nz))
    alpha = 0.5  # starts with moderate DK reuse and can adapt

    z_state = np.zeros(nz, dtype=np.complex128)

    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    reward_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    p_trace = np.zeros(cfg.num_slots, dtype=np.float64)
    leak_trace = np.zeros(cfg.num_slots, dtype=np.float64)

    dk_var = rl_cfg.sigma_dk2 * np.ones(action_dim, dtype=np.float64)
    batch_size = max(int(rl_cfg.batch_size), 1)

    batch_grad_w_mu = np.zeros_like(w_mu)
    batch_grad_u_sigma = np.zeros_like(u_sigma)
    batch_grad_alpha = 0.0
    batch_count = 0

    for t in range(cfg.num_slots):

        print("RL Slot {} / {}".format(t + 1, cfg.num_slots), end="\r")

        # ----- State = PMI feedback -----
        y = zf_baseline["pmi_features"][t]
        z_state = split_tanh(win @ y + wres @ z_state)

        # Learned Gaussian policy parameters.
        mu = w_mu @ z_state
        z_aug = np.concatenate([np.real(z_state), np.imag(z_state)], axis=0)
        ell = u_sigma @ z_aug
        sigma2 = softplus(ell) + rl_cfg.sigma_eps

        # DK branch centered at ZF precoder of this slot.
        dk_mu = zf_baseline["precoders"][t].reshape(-1)

        # Mixture policy parameter p.
        p = float(sigmoid(alpha))
        p = np.clip(p, 1e-6, 1 - 1e-6)

        # Sample action from mixture of two complex Gaussians.
        if rng.random() < p:
            action_vec_raw = sample_complex_gaussian(mu, sigma2, rng)
        else:
            action_vec_raw = sample_complex_gaussian(dk_mu, dk_var, rng)

        # Hard constraint: total transmit power normalization.
        action_precoder = action_vec_raw.reshape(cfg.num_tx_antennas, num_streams)
        action_precoder = normalize_precoder_frobenius(action_precoder, cfg.total_tx_power)

        # Environment throughput and leakage.
        rate_rl, _ = compute_slot_sum_rate(channels[t], action_precoder, noise_power)
        leakage = compute_leakage(channels[t], action_precoder)
        reward = (rate_rl - zf_baseline["throughput"][t]) - rl_cfg.leakage_penalty_weight * leakage

        # Use the normalized action for policy-gradient likelihood as well.
        action_vec = action_precoder.reshape(-1)

        # Mixture log-density terms.
        log_pi_phi = complex_diag_logpdf(action_vec, mu, sigma2)
        log_pi_dk = complex_diag_logpdf(action_vec, dk_mu, dk_var)

        log_num_phi = np.log(p) + log_pi_phi
        log_num_dk = np.log(1 - p) + log_pi_dk
        m = max(log_num_phi, log_num_dk)
        log_pi_theta = m + np.log(np.exp(log_num_phi - m) + np.exp(log_num_dk - m))

        # Responsibility weight w_t = p*pi_phi / pi_theta.
        w_t = np.exp(log_num_phi - log_pi_theta)

        # ----- Policy gradients from the LaTeX derivation -----
        err = action_vec - mu
        inv_sigma2 = 1.0 / sigma2

        # d/dW_mu* log pi_theta = w_t * Sigma^-1 (a-mu) z^H
        grad_w_mu_log = w_t * np.outer(inv_sigma2 * err, np.conj(z_state))

        # d/dU_sigma log pi_theta = w_t * delta * z_aug^T
        abs_err2 = np.abs(err) ** 2
        delta = ((-inv_sigma2) + (abs_err2 * (inv_sigma2**2))) * sigmoid(ell)
        grad_u_sigma_log = w_t * np.outer(delta, z_aug)

        # d/dalpha log pi_theta = p(1-p) * (pi_phi - pi_dk)/pi_theta
        # Numerically stable equivalent:
        # (pi_phi/pi_theta) = w_t/p, (pi_dk/pi_theta) = (1-w_t)/(1-p)
        dlog_dp = (w_t / p) - ((1 - w_t) / (1 - p))
        grad_alpha_log = p * (1 - p) * dlog_dp

        # Reward-weighted policy-gradient contributions (advantage = reward).
        grad_w_mu_contrib = reward * grad_w_mu_log
        grad_u_sigma_contrib = reward * grad_u_sigma_log
        grad_alpha_contrib = reward * grad_alpha_log

        batch_grad_w_mu += grad_w_mu_contrib
        batch_grad_u_sigma += grad_u_sigma_contrib
        batch_grad_alpha += grad_alpha_contrib
        batch_count += 1

        # Apply averaged batch update every B slots.
        if batch_count == batch_size:
            scale = 1.0 / batch_count
            w_mu += rl_cfg.lr_mu * scale * batch_grad_w_mu
            u_sigma += rl_cfg.lr_sigma * scale * batch_grad_u_sigma
            alpha += rl_cfg.lr_alpha * scale * batch_grad_alpha

            batch_grad_w_mu.fill(0.0)
            batch_grad_u_sigma.fill(0.0)
            batch_grad_alpha = 0.0
            batch_count = 0
        throughput[t] = rate_rl
        reward_trace[t] = reward
        p_trace[t] = p
        leak_trace[t] = leakage

    # Flush last partial batch.
    if batch_count > 0:
        scale = 1.0 / batch_count
        w_mu += rl_cfg.lr_mu * scale * batch_grad_w_mu
        u_sigma += rl_cfg.lr_sigma * scale * batch_grad_u_sigma
        alpha += rl_cfg.lr_alpha * scale * batch_grad_alpha

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
    rl_trace = output_dir / "hybrid_rl_throughput_trace.npy"
    reward_trace = output_dir / "hybrid_rl_reward_trace.npy"
    learned_policy_prob_trace = output_dir / "hybrid_rl_learned_policy_probability_trace.npy"
    domain_knowledge_prob_trace = output_dir / "hybrid_rl_domain_knowledge_probability_trace.npy"


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
    ax1.plot(np.arange(1, len(rl_tput_avg) + 1), rl_tput_avg, lw=1.6, label="Hybrid RL")
    ax1.set_title("Throughput Across Time")
    ax1.set_xlabel("Slot index")
    ax1.set_ylabel("Sum-rate [bits/s/Hz]")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")
    fig1.tight_layout()

    tput_plot = output_dir / "throughput_across_time_zf_vs_hybrid_rl.png"
    fig1.savefig(tput_plot, dpi=150)
    plt.close(fig1)

    # Reward plot
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(np.arange(1, len(reward_avg) + 1), reward_avg, lw=1.6)
    ax2.set_title("Hybrid RL Reward Across Time\n(reward = throughput gain over ZF - leakage penalty)")
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
    ax3.set_title("Hybrid RL Branch Probabilities Across Time")
    ax3.set_xlabel("Slot index")
    ax3.set_ylabel("Probability")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.35)
    ax3.legend(loc="best")
    fig3.tight_layout()

    probability_plot = output_dir / "hybrid_rl_branch_probabilities_across_time.png"
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
    print(f"Hybrid RL average throughput: {rl_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Hybrid RL average reward    : {rl_results['reward'].mean():.4f}")
    print("Saved artifacts:")
    for name, path in out.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()