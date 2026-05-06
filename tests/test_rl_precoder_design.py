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
    beta_zf: float = 2.0
    kappa_min: float = 1e-3
    kappa_dk: float = 50.0
    lr_mu: float = 3e-4
    lr_kappa: float = 1e-4
    lr_alpha: float = 2e-3
    batch_size: int = 50
    rho_gamma: float = 0.7
    max_resamples: int = 8

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
    selected_rows = []
    for vk in full_vk_list:
        vk_dom = vk[:, :streams_per_user]
        selected_rows.append(vk_dom.conj().T)

    z_matrix = np.vstack(selected_rows)
    precoder = np.linalg.pinv(z_matrix)
    return normalize_columns_equal_power(precoder, total_tx_power)


def compute_slot_sum_rate(user_channels: np.ndarray, precoder: np.ndarray, noise_power: float) -> tuple[float, np.ndarray]:
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

def pmi_features_from_channels(user_channels: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
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
    channels = np.zeros((cfg.num_slots, cfg.num_users, cfg.num_rx_antennas_per_user, cfg.num_tx_antennas), dtype=np.complex128)

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

    return {"throughput": throughput, "sinr": sinr_traces, "precoders": precoders, "pmi_features": np.stack(pmi_features, axis=0)}


def real_to_complex_beam(x: np.ndarray, total_tx_power: float, num_users: int) -> np.ndarray:
    nt = x.size // 2
    return np.sqrt(total_tx_power / num_users) * (x[:nt] + 1j * x[nt:])

def complex_to_real_unit(p: np.ndarray, total_tx_power: float, num_users: int) -> np.ndarray:
    scale = np.sqrt(total_tx_power / num_users)
    x = np.concatenate([np.real(p), np.imag(p)], axis=0) / max(scale, 1e-12)
    n = np.linalg.norm(x)
    return x / max(n, 1e-12)


def sample_vmf(mu: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal(mu.shape)
    y = kappa * mu + z
    n = np.linalg.norm(y)
    return y / max(n, 1e-12)


def log_c_approx(d: int, kappa: float) -> float:
    return (d / 2.0 - 1.0) * np.log(max(kappa, 1e-12)) - (d / 2.0) * np.log(2 * np.pi) - kappa


def log_vmf(x: np.ndarray, mu: np.ndarray, kappa: float) -> float:
    return float(log_c_approx(x.size, kappa) + kappa * np.dot(mu, x))


def run_hybrid_rl(cfg: SimConfig, rl_cfg: RLConfig, channels: np.ndarray, zf_baseline: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))
    pmi_dim = zf_baseline["pmi_features"].shape[1]
    nz = rl_cfg.reservoir_size
    k = cfg.num_users
    d = 2 * cfg.num_tx_antennas

    # Fixed ESN reservoir.
    win = rl_cfg.input_scale * complex_gaussian((nz, pmi_dim), rng)
    wres = complex_gaussian((nz, nz), rng)
    eigvals = np.linalg.eigvals(wres)
    wres = (rl_cfg.spectral_radius / max(np.max(np.abs(eigvals)), 1e-12)) * wres

    # Trainable readout parameters.
    w_mu = 1e-3 * rng.standard_normal((k, d, 2 * nz))
    u_kappa = 1e-3 * rng.standard_normal((k, 2 * nz))
    alpha = 1.0

    z_state = np.zeros(nz, dtype=np.complex128)

    throughput = np.zeros(cfg.num_slots)
    reward_trace = np.zeros(cfg.num_slots)
    p_trace = np.zeros(cfg.num_slots)
    acceptance_trace = np.zeros(cfg.num_slots)
    fallback_trace = np.zeros(cfg.num_slots)
    attempts_trace = np.zeros(cfg.num_slots)
    sinr_ratio_trace = np.zeros((cfg.num_slots, k))

    bg_w_mu = np.zeros_like(w_mu)
    bg_u_kappa = np.zeros_like(u_kappa)
    bg_alpha = 0.0
    bcount = 0

    for t in range(cfg.num_slots):

        print("RL Slot {} / {}".format(t + 1, cfg.num_slots), end="\r")

        # ----- State = PMI feedback -----
        y = zf_baseline["pmi_features"][t]
        z_state = split_tanh(win @ y + wres @ z_state)

        # Learned Gaussian policy parameters.
        z_aug = np.concatenate([np.real(z_state), np.imag(z_state)])
        p_zf = normalize_columns_equal_power(zf_baseline["precoders"][t], cfg.total_tx_power)
        _, sinr_zf = compute_slot_sum_rate(channels[t], p_zf, noise_power)

        mu_zf = np.zeros((k, d))
        for ku in range(k):
            mu_zf[ku] = complex_to_real_unit(p_zf[:, ku], cfg.total_tx_power, k)

        h = np.zeros((k, d))
        mu_phi = np.zeros((k, d))
        kappa_phi = np.zeros(k)
        c_k = np.zeros(k)
        for ku in range(k):
            h[ku] = rl_cfg.beta_zf * mu_zf[ku] + w_mu[ku] @ z_aug
            h_n = np.linalg.norm(h[ku])
            mu_phi[ku] = h[ku] / max(h_n, 1e-12)
            c_k[ku] = u_kappa[ku] @ z_aug
            kappa_phi[ku] = float(softplus(np.array([c_k[ku]]))[0] + rl_cfg.kappa_min)

        p = float(np.clip(sigmoid(alpha), 1e-6, 1 - 1e-6))
        p_trace[t] = p

        accepted = False
        chosen = None
        chosen_x = None
        chosen_from_phi = False
        for m in range(1, rl_cfg.max_resamples + 1):
            from_phi = rng.random() < p
            xs = np.zeros((k, d))
            for ku in range(k):
                if from_phi:
                    xs[ku] = sample_vmf(mu_phi[ku], kappa_phi[ku], rng)
                else:
                    xs[ku] = sample_vmf(mu_zf[ku], rl_cfg.kappa_dk, rng)

            p_cand = np.zeros((cfg.num_tx_antennas, k), dtype=np.complex128)
            for ku in range(k):
                p_cand[:, ku] = real_to_complex_beam(xs[ku], cfg.total_tx_power, k)

            _, sinr_cand = compute_slot_sum_rate(channels[t], p_cand, noise_power)
            ratios = sinr_cand / np.maximum(sinr_zf, 1e-12)
            if np.all(sinr_cand >= rl_cfg.rho_gamma * sinr_zf):
                accepted = True
                chosen = p_cand
                chosen_x = xs
                chosen_from_phi = from_phi
                attempts_trace[t] = m
                sinr_ratio_trace[t] = ratios
                break

        if not accepted:
            chosen = p_zf
            attempts_trace[t] = rl_cfg.max_resamples
            fallback_trace[t] = 1.0
            _, sinr_exec = compute_slot_sum_rate(channels[t], chosen, noise_power)
            sinr_ratio_trace[t] = sinr_exec / np.maximum(sinr_zf, 1e-12)

        rate_exec, _ = compute_slot_sum_rate(channels[t], chosen, noise_power)
        reward = rate_exec - zf_baseline["throughput"][t]
        throughput[t] = rate_exec
        reward_trace[t] = reward
        acceptance_trace[t] = 1.0 if accepted else 0.0

        if accepted and reward != 0.0:
            log_pi_phi = 0.0
            log_pi_dk = 0.0
            for ku in range(k):
                xk = chosen_x[ku]
                log_pi_phi += log_vmf(xk, mu_phi[ku], kappa_phi[ku])
                log_pi_dk += log_vmf(xk, mu_zf[ku], rl_cfg.kappa_dk)

            lphi = np.log(p) + log_pi_phi
            ldk = np.log(1 - p) + log_pi_dk
            m = max(lphi, ldk)
            log_q = m + np.log(np.exp(lphi - m) + np.exp(ldk - m))
            w_t = np.exp(lphi - log_q)

            for ku in range(k):
                xk = chosen_x[ku]
                hk = h[ku]
                mu_k = mu_phi[ku]
                hk_norm = np.linalg.norm(hk)
                proj = np.eye(d) - np.outer(mu_k, mu_k)
                grad_h = (kappa_phi[ku] / max(hk_norm, 1e-12)) * (proj @ xk)
                bg_w_mu[ku] += reward * w_t * np.outer(grad_h, z_aug)

                a_d = 1.0 - (d - 1) / max(2.0 * kappa_phi[ku], 1e-6)
                grad_c = ((mu_k @ xk) - a_d) * sigmoid(c_k[ku])
                bg_u_kappa[ku] += reward * w_t * grad_c * z_aug

            bg_alpha += reward * (w_t - p)
            bcount += 1

        if bcount == rl_cfg.batch_size:
            s = 1.0 / bcount
            w_mu += rl_cfg.lr_mu * s * bg_w_mu
            u_kappa += rl_cfg.lr_kappa * s * bg_u_kappa
            alpha += rl_cfg.lr_alpha * s * bg_alpha
            bg_w_mu.fill(0.0)
            bg_u_kappa.fill(0.0)
            bg_alpha = 0.0
            bcount = 0

    if bcount > 0:
        s = 1.0 / bcount
        w_mu += rl_cfg.lr_mu * s * bg_w_mu
        u_kappa += rl_cfg.lr_kappa * s * bg_u_kappa
        alpha += rl_cfg.lr_alpha * s * bg_alpha

    return {
        "throughput": throughput,
        "reward": reward_trace,
        "p_trace": p_trace,
        "acceptance": acceptance_trace,
        "fallback": fallback_trace,
        "attempts": attempts_trace,
        "sinr_ratio": sinr_ratio_trace,
    }


def save_results(zf_results: dict[str, np.ndarray], rl_results: dict[str, np.ndarray], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "zf_throughput_trace.npy", zf_results["throughput"])
    np.save(output_dir / "hybrid_rl_throughput_trace.npy", rl_results["throughput"])
    np.save(output_dir / "hybrid_rl_reward_trace.npy", rl_results["reward"])
    np.save(output_dir / "hybrid_rl_learned_policy_probability_trace.npy", rl_results["p_trace"])
    np.save(output_dir / "hybrid_rl_acceptance_trace.npy", rl_results["acceptance"])
    np.save(output_dir / "hybrid_rl_fallback_trace.npy", rl_results["fallback"])
    np.save(output_dir / "hybrid_rl_attempts_trace.npy", rl_results["attempts"])
    np.save(output_dir / "hybrid_rl_sinr_ratio_trace.npy", rl_results["sinr_ratio"])

    zf_tput = zf_results["throughput"]
    rl_tput = rl_results["throughput"]

    # Save traces
    zf_trace = output_dir / "zf_throughput_trace.npy"
    rl_trace = output_dir / "hybrid_rl_throughput_trace.npy"
    reward_trace = output_dir / "hybrid_rl_reward_trace.npy"
    learned_policy_prob_trace = output_dir / "hybrid_rl_learned_policy_probability_trace.npy"
    domain_knowledge_prob_trace = output_dir / "hybrid_rl_domain_knowledge_probability_trace.npy"

    w_len = 10000
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
    parser.add_argument("--num-slots", type=int, default=100000)
    parser.add_argument("--snr-db", type=float, default=15.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reservoir-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("results/simple_zf_hybrid_rl_testbench"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = SimConfig(num_slots=args.num_slots, snr_db=args.snr_db, temporal_correlation=args.rho, seed=args.seed)
    rl_cfg = RLConfig(reservoir_size=args.reservoir_size)

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
    save_results(zf_results, rl_results, args.output_dir)


if __name__ == "__main__":
    main()