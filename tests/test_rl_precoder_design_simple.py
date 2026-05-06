from __future__ import annotations

import argparse
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
    lr_mu: float = 3e-3
    lr_kappa: float = 1e-3
    kappa_min: float = 1e-3


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


def build_zf_precoder_from_pmi(full_vk_list: list[np.ndarray], streams_per_user: int, total_tx_power: float) -> np.ndarray:
    selected_rows = []
    for vk in full_vk_list:
        selected_rows.append(vk[:, :streams_per_user].conj().T)
    z_matrix = np.vstack(selected_rows)
    return normalize_columns_equal_power(np.linalg.pinv(z_matrix), total_tx_power)


def compute_slot_sum_rate(user_channels: np.ndarray, precoder: np.ndarray, noise_power: float) -> tuple[float, np.ndarray]:
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
    channels = np.zeros((cfg.num_slots, cfg.num_users, cfg.num_rx_antennas_per_user, cfg.num_tx_antennas), dtype=np.complex128)
    h_prev = complex_gaussian((cfg.num_users, cfg.num_rx_antennas_per_user, cfg.num_tx_antennas), rng)
    for t in range(cfg.num_slots):
        innovation = complex_gaussian(h_prev.shape, rng)
        h_prev = cfg.temporal_correlation * h_prev + np.sqrt(1 - cfg.temporal_correlation**2) * innovation
        channels[t] = h_prev
    return channels


def run_zf_baseline(cfg: SimConfig, channels: np.ndarray) -> dict[str, np.ndarray]:
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    pmi_features = []
    for t in range(cfg.num_slots):
        print(f"ZF Slot {t + 1} / {cfg.num_slots}", end="\r")
        feat, vk_list = pmi_features_from_channels(channels[t])
        pmi_features.append(feat)
        p_zf = build_zf_precoder_from_pmi(vk_list, cfg.streams_per_user, cfg.total_tx_power)
        throughput[t], _ = compute_slot_sum_rate(channels[t], p_zf, noise_power)
    print()
    return {"throughput": throughput, "pmi_features": np.stack(pmi_features, axis=0)}


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def unit_norm(x: np.ndarray) -> np.ndarray:
    return x / max(np.linalg.norm(x), 1e-12)


def real_to_complex_beam(x: np.ndarray, total_tx_power: float, num_users: int) -> np.ndarray:
    nt = x.size // 2
    return np.sqrt(total_tx_power / num_users) * (x[:nt] + 1j * x[nt:])

def run_random_vmf_baseline(cfg: SimConfig, channels: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    k = cfg.num_users
    d = 2 * cfg.num_tx_antennas

    for t in range(cfg.num_slots):
        print(f"Random vMF Slot {t + 1} / {cfg.num_slots}", end="\r")
        beams = np.zeros((cfg.num_tx_antennas, k), dtype=np.complex128)
        for ku in range(k):
            mu = unit_norm(rng.standard_normal(d))
            kappa = float(rng.uniform(1e-3, 10.0))
            x_sample = sample_vmf(mu, kappa, rng)
            beams[:, ku] = real_to_complex_beam(x_sample, cfg.total_tx_power, k)
        throughput[t], _ = compute_slot_sum_rate(channels[t], beams, noise_power)

    print()
    return {"throughput": throughput}

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


def vmf_log_prob_torch(x: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    # Constant term omitted since it does not affect policy gradients.
    return kappa * torch.sum(mu * x)


def run_single_policy_rl(cfg: SimConfig, rl_cfg: RLConfig, channels: np.ndarray, zf_baseline: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    feat_dim = zf_baseline["pmi_features"].shape[1]
    k = cfg.num_users
    d = 2 * cfg.num_tx_antennas

    # Single learned policy, random init (no DK dependency in parameterization).
    w_mu = 1e-2 * rng.standard_normal((k, d, feat_dim))
    w_kappa = 1e-2 * rng.standard_normal((k, feat_dim))

    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    reward_trace = np.zeros(cfg.num_slots, dtype=np.float64)

    for t in range(cfg.num_slots):
        print(f"RL Slot {t + 1} / {cfg.num_slots}", end="\r")
        y = zf_baseline["pmi_features"][t]
        beams = np.zeros((cfg.num_tx_antennas, k), dtype=np.complex128)

        mu = np.zeros((k, d), dtype=np.float64)
        kappa = np.zeros(k, dtype=np.float64)
        x_sample = np.zeros((k, d), dtype=np.float64)

        for ku in range(k):
            mu[ku] = unit_norm(w_mu[ku] @ y)
            kappa[ku] = float(softplus(np.array([w_kappa[ku] @ y]))[0] + rl_cfg.kappa_min)
            x_sample[ku] = sample_vmf(mu[ku], kappa[ku], rng)
            beams[:, ku] = real_to_complex_beam(x_sample[ku], cfg.total_tx_power, k)

        rate, _ = compute_slot_sum_rate(channels[t], beams, noise_power)
        reward = rate - zf_baseline["throughput"][t]

        throughput[t] = rate
        reward_trace[t] = reward

        # REINFORCE update using PyTorch autograd and vMF log-probability.
        y_t = torch.tensor(y, dtype=torch.float64)
        for ku in range(k):
            w_mu_t = torch.tensor(w_mu[ku], dtype=torch.float64, requires_grad=True)
            w_kappa_t = torch.tensor(w_kappa[ku], dtype=torch.float64, requires_grad=True)
            x_t = torch.tensor(x_sample[ku], dtype=torch.float64)

            mu_t = w_mu_t @ y_t
            mu_t = mu_t / torch.clamp(torch.linalg.norm(mu_t), min=1e-12)
            kappa_t = torch.nn.functional.softplus(torch.dot(w_kappa_t, y_t)) + rl_cfg.kappa_min
            log_prob = vmf_log_prob_torch(x_t, mu_t, kappa_t)
            loss = -reward * log_prob
            loss.backward()

            with torch.no_grad():
                w_mu[ku] -= rl_cfg.lr_mu * w_mu_t.grad.detach().numpy()
                w_kappa[ku] -= rl_cfg.lr_kappa * w_kappa_t.grad.detach().numpy()

    print()
    return {"throughput": throughput, "reward": reward_trace}


def moving_average(trace: np.ndarray, window_len: int) -> np.ndarray:
    if window_len <= 1 or window_len > trace.size:
        return trace.copy()
    kernel = np.ones(window_len, dtype=np.float64) / window_len
    return np.convolve(trace, kernel, mode="valid")


def save_plots(zf_throughput: np.ndarray, random_vmf_throughput: np.ndarray, rl_throughput: np.ndarray, reward: np.ndarray, output_dir: Path, window_len: int) -> None:
    zf_avg = moving_average(zf_throughput, window_len)
    random_vmf_avg = moving_average(random_vmf_throughput, window_len)
    rl_avg = moving_average(rl_throughput, window_len)
    reward_avg = moving_average(reward, window_len)

    x_tput = np.arange(1, zf_avg.size + 1)
    x_reward = np.arange(1, reward_avg.size + 1)

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x_tput, zf_avg, lw=1.5, label="ZF baseline")
    ax1.plot(x_tput, random_vmf_avg, lw=1.5, label="Random vMF baseline")
    ax1.plot(x_tput, rl_avg, lw=1.5, label="Single-policy RL")
    ax1.set_title("Throughput Across Time")
    ax1.set_xlabel("Slot index")
    ax1.set_ylabel("Sum-rate [bits/s/Hz]")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(output_dir / "throughput_across_time_zf_vs_single_policy_rl.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(x_reward, reward_avg, lw=1.5)
    ax2.set_title("Single-policy RL Reward Across Time")
    ax2.set_xlabel("Slot index")
    ax2.set_ylabel("Reward vs ZF")
    ax2.grid(True, alpha=0.35)
    fig2.tight_layout()
    fig2.savefig(output_dir / "single_policy_rl_reward_across_time.png", dpi=150)
    plt.close(fig2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple MU-MIMO ZF baseline + single-policy RL")
    parser.add_argument("--num-slots", type=int, default=100000)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/simple_rl_precoder_design"))
    parser.add_argument("--window-len", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(num_slots=args.num_slots, snr_db=args.snr_db, temporal_correlation=args.rho, seed=args.seed)
    rl_cfg = RLConfig()

    rng = np.random.default_rng(cfg.seed)
    channels = simulate_channels(cfg, rng)

    # keep DK as comparison baseline only
    zf_results = run_zf_baseline(cfg, channels)
    random_vmf_results = run_random_vmf_baseline(cfg, channels, rng)
    rl_results = run_single_policy_rl(cfg, rl_cfg, channels, zf_results, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "zf_throughput_trace.npy", zf_results["throughput"])
    np.save(args.output_dir / "random_vmf_baseline_throughput_trace.npy", random_vmf_results["throughput"])
    np.save(args.output_dir / "single_policy_rl_throughput_trace.npy", rl_results["throughput"])
    np.save(args.output_dir / "single_policy_rl_reward_trace.npy", rl_results["reward"])

    save_plots(
        zf_throughput=zf_results["throughput"],
        random_vmf_throughput=random_vmf_results["throughput"],
        rl_throughput=rl_results["throughput"],
        reward=rl_results["reward"],
        output_dir=args.output_dir,
        window_len=args.window_len,
    )

    print("Simple RL precoder design run finished.")
    print(f"ZF average throughput         : {zf_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Random vMF baseline throughput: {random_vmf_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Single-policy RL throughput   : {rl_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Single-policy RL reward       : {rl_results['reward'].mean():.4f}")


if __name__ == "__main__":
    main()