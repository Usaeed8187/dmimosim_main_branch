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

        per_user_log_prob = vmf_log_prob_fixed_kappa_torch(x_batch, mu, rl_cfg.fixed_kappa)
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

        beams = np.zeros((cfg.num_tx_antennas, k), dtype=np.complex128)
        x_sample = np.zeros((k, d), dtype=np.float64)
        for ku in range(k):
            x_sample[ku] = sample_vmf(mu_np[ku], rl_cfg.fixed_kappa, rng)
            beams[:, ku] = real_to_complex_beam(x_sample[ku], cfg.total_tx_power, k)

        rate, _ = compute_slot_sum_rate(channels[t], beams, noise_power)
        reward = rate - zf_baseline["throughput"][t]

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
        advantage_trace[t] = advantage
        baseline_trace[t] = reward_baseline
        beat_zf_trace[t] = 1.0 if rate > zf_baseline["throughput"][t] else 0.0
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
        "advantage": advantage_trace,
        "reward_baseline": baseline_trace,
        "kappa": kappa_trace,
        "beat_zf": beat_zf_trace,
        "beam_similarity_to_zf": beam_similarity_trace,
        "grad_norm": grad_norm_trace,
        "loss": loss_trace,
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

    zf_avg = moving_average(zf_throughput, window_len)
    random_vmf_avg = moving_average(random_vmf_throughput, window_len)
    rl_avg = moving_average(rl_throughput, window_len)
    reward_avg = moving_average(reward, window_len)
    beat_zf_avg = moving_average(beat_zf, window_len)
    sim_avg = moving_average(sim_to_zf, window_len)
    grad_norm_avg = moving_average(grad_norm, window_len)

    x_tput = np.arange(1, zf_avg.size + 1)
    x_reward = np.arange(1, reward_avg.size + 1)

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x_tput, zf_avg, lw=1.5, label="ZF baseline")
    ax1.plot(x_tput, random_vmf_avg, lw=1.5, label="Random vMF baseline")
    ax1.plot(x_tput, rl_avg, lw=1.5, label="ESN-vMF RL")
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
    ax2.set_title("ESN-vMF RL Reward Across Time")
    ax2.set_xlabel("Slot index")
    ax2.set_ylabel("Reward vs ZF")
    ax2.grid(True, alpha=0.35)
    fig2.tight_layout()
    fig2.savefig(output_dir / "esn_vmf_rl_reward_across_time.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.plot(np.arange(1, beat_zf_avg.size + 1), beat_zf_avg, lw=1.5)
    ax3.set_title("Fraction of Slots Where ESN-vMF RL Beats ZF")
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
    fig5.savefig(output_dir / "esn_vmf_rl_grad_norm.png", dpi=150)
    plt.close(fig5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple MU-MIMO ZF baseline + batched ESN-vMF RL")
    parser.add_argument("--num-slots", type=int, default=100000)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/simple_esn_vmf_rl_precoder_design"))
    parser.add_argument("--window-len", type=int, default=1000)

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
    )

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    channels = simulate_channels(cfg, rng)

    zf_results = run_zf_baseline(cfg, channels)
    random_vmf_results = run_random_vmf_baseline(
        cfg, channels, rng, fixed_kappa=rl_cfg.fixed_kappa
    )
    rl_results = run_esn_policy_rl(cfg, rl_cfg, channels, zf_results, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "zf_throughput_trace.npy", zf_results["throughput"])
    np.save(args.output_dir / "random_vmf_baseline_throughput_trace.npy", random_vmf_results["throughput"])
    np.save(args.output_dir / "esn_vmf_rl_throughput_trace.npy", rl_results["throughput"])
    np.save(args.output_dir / "esn_vmf_rl_reward_trace.npy", rl_results["reward"])
    np.save(args.output_dir / "esn_vmf_rl_advantage_trace.npy", rl_results["advantage"])
    np.save(args.output_dir / "esn_vmf_rl_reward_baseline_trace.npy", rl_results["reward_baseline"])
    np.save(args.output_dir / "esn_vmf_rl_kappa_trace.npy", rl_results["kappa"])
    np.save(args.output_dir / "esn_vmf_rl_fraction_beats_zf_trace.npy", rl_results["beat_zf"])
    np.save(args.output_dir / "esn_vmf_rl_beam_similarity_to_zf_trace.npy", rl_results["beam_similarity_to_zf"])
    np.save(args.output_dir / "esn_vmf_rl_grad_norm_trace.npy", rl_results["grad_norm"])
    np.save(args.output_dir / "esn_vmf_rl_loss_trace.npy", rl_results["loss"])
    np.save(args.output_dir / "esn_states_trace.npy", rl_results["esn_states"])

    save_plots(
        zf_throughput=zf_results["throughput"],
        random_vmf_throughput=random_vmf_results["throughput"],
        rl_results=rl_results,
        output_dir=args.output_dir,
        window_len=args.window_len,
    )

    print("Simple ESN-vMF RL precoder design run finished.")
    print(f"ZF average throughput         : {zf_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Random vMF baseline throughput: {random_vmf_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"ESN-vMF RL throughput         : {rl_results['throughput'].mean():.4f} bits/s/Hz")
    print(f"ESN-vMF RL reward             : {rl_results['reward'].mean():.4f}")
    print(f"RL beats ZF fraction          : {rl_results['beat_zf'].mean():.4f}")
    print(f"Mean beam similarity to ZF    : {rl_results['beam_similarity_to_zf'].mean():.4f}")
    print(f"Mean grad norm                : {rl_results['grad_norm'].mean():.4f}")


if __name__ == "__main__":
    main()
