"""Simple MU-MIMO ZF testbench using ideal PMI from full right singular vectors.

This script intentionally avoids dMIMO pipeline features and reinforcement learning.
It simulates a single-cell setup with:
  - N_t = 4 transmit antennas
  - K = 2 users
  - N_r = 2 receive antennas per user
  - 1 stream per user

At each slot, channels evolve with a Gauss-Markov process. For each user, we compute
an SVD of H_k and use the full right singular-vector matrix V_k as PMI feedback
(with no quantization/noise). The dominant right singular vector of each user is
stacked to build the ZF precoder.

The script tracks and plots throughput over time.
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


def complex_gaussian(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Generate circularly-symmetric complex Gaussian samples with unit variance."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def normalize_precoder_frobenius(precoder: np.ndarray, total_tx_power: float) -> np.ndarray:
    """Scale precoder to satisfy total transmit power."""
    fro_norm = np.linalg.norm(precoder, ord="fro")
    if fro_norm < 1e-12:
        return precoder
    return np.sqrt(total_tx_power) * precoder / fro_norm


def build_zf_precoder_from_pmi(full_vk_list: list[np.ndarray], streams_per_user: int, total_tx_power: float) -> np.ndarray:
    """Build ZF precoder from user PMI matrices V_k.

    Args:
        full_vk_list: List of full right singular-vector matrices V_k (N_t x N_t).
        streams_per_user: Number of streams per user.
        total_tx_power: Total transmit power for Frobenius normalization.

    Returns:
        Precodermatrix P of shape (N_t, K*streams_per_user).
    """
    selected_rows = []
    for vk in full_vk_list:
        # Use first "streams_per_user" dominant right singular vectors.
        vk_dom = vk[:, :streams_per_user]  # (N_t, d_k)
        selected_rows.append(vk_dom.conj().T)  # (d_k, N_t)

    z_matrix = np.vstack(selected_rows)  # (N_s, N_t), N_s = K*streams_per_user
    # Moore-Penrose pseudoinverse for ZF.
    precoder = np.linalg.pinv(z_matrix)  # (N_t, N_s)
    return normalize_precoder_frobenius(precoder, total_tx_power=total_tx_power)


def compute_slot_sum_rate(
    user_channels: np.ndarray,
    precoder: np.ndarray,
    noise_power: float,
) -> tuple[float, np.ndarray]:
    """Compute sum-rate and per-user SINR for one slot.

    For user k (one stream), we use a matched receive combiner based on the
    intended effective spatial signature g_{k,k} = H_k p_k.
    """
    num_users = user_channels.shape[0]
    sinr = np.zeros(num_users, dtype=np.float64)

    for k in range(num_users):
        hk = user_channels[k]  # (N_r, N_t)

        signal_vec = hk @ precoder[:, k]
        signal_norm = np.linalg.norm(signal_vec)
        if signal_norm < 1e-12:
            uk = np.zeros_like(signal_vec)
            uk[0] = 1.0 + 0j
        else:
            uk = signal_vec / signal_norm

        desired = np.abs(np.vdot(uk, hk @ precoder[:, k])) ** 2

        interference = 0.0
        for j in range(num_users):
            if j == k:
                continue
            interference += np.abs(np.vdot(uk, hk @ precoder[:, j])) ** 2

        sinr[k] = desired / (interference + noise_power)

    sum_rate = float(np.sum(np.log2(1.0 + sinr)))
    return sum_rate, sinr


def run_simulation(cfg: SimConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    if cfg.streams_per_user != 1:
        raise ValueError("This simple testbench currently supports streams_per_user = 1.")

    num_streams = cfg.num_users * cfg.streams_per_user
    noise_power = cfg.total_tx_power / (10.0 ** (cfg.snr_db / 10.0))

    # Initial channels: shape (K, N_r, N_t)
    channels = complex_gaussian(
        (cfg.num_users, cfg.num_rx_antennas_per_user, cfg.num_tx_antennas),
        rng,
    )

    throughput = np.zeros(cfg.num_slots, dtype=np.float64)
    sinr_traces = np.zeros((cfg.num_slots, cfg.num_users), dtype=np.float64)

    for t in range(cfg.num_slots):
        # Temporal evolution with Gauss-Markov fading.
        innovation = complex_gaussian(channels.shape, rng)
        channels = cfg.temporal_correlation * channels + np.sqrt(1 - cfg.temporal_correlation**2) * innovation

        # PMI feedback: full right singular vectors V_k from SVD(H_k).
        full_vk_list = []
        for k in range(cfg.num_users):
            _, _, vh = np.linalg.svd(channels[k], full_matrices=True)
            full_vk_list.append(vh.conj().T)

        # ZF precoder from PMI.
        precoder = build_zf_precoder_from_pmi(
            full_vk_list=full_vk_list,
            streams_per_user=cfg.streams_per_user,
            total_tx_power=cfg.total_tx_power,
        )

        if precoder.shape != (cfg.num_tx_antennas, num_streams):
            raise RuntimeError(
                f"Unexpected precoder shape {precoder.shape}, expected {(cfg.num_tx_antennas, num_streams)}"
            )

        slot_sum_rate, slot_sinr = compute_slot_sum_rate(
            user_channels=channels,
            precoder=precoder,
            noise_power=noise_power,
        )

        throughput[t] = slot_sum_rate
        sinr_traces[t] = slot_sinr

    return {
        "throughput": throughput,
        "sinr": sinr_traces,
    }


def save_results(results: dict[str, np.ndarray], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    throughput = results["throughput"]

    # Save numeric trace.
    trace_path = output_dir / "zf_throughput_trace.npy"
    np.save(trace_path, throughput)

    # Save plot.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(throughput) + 1), throughput, lw=1.8, label="ZF sum-rate")
    ax.set_title("Throughput Across Time (Simple ZF + Full PMI)")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Sum-rate [bits/s/Hz]")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = output_dir / "zf_throughput_across_time.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return trace_path, plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple MU-MIMO ZF testbench with full PMI feedback")
    parser.add_argument("--num-slots", type=int, default=300, help="Number of simulated time slots")
    parser.add_argument("--snr-db", type=float, default=15.0, help="SNR in dB")
    parser.add_argument("--rho", type=float, default=0.95, help="Temporal channel correlation coefficient")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simple_zf_testbench"),
        help="Directory for saved traces and plot",
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

    results = run_simulation(cfg)
    trace_path, plot_path = save_results(results, args.output_dir)

    print("Simple ZF testbench finished.")
    print(f"Average throughput: {results['throughput'].mean():.4f} bits/s/Hz")
    print(f"Saved throughput trace: {trace_path}")
    print(f"Saved throughput plot : {plot_path}")


if __name__ == "__main__":
    main()