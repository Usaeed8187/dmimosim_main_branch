#!/usr/bin/env python3
"""Plot and quantify temporal variation of an ns-3 TxSquad channel.

This reads the saved npz files directly and does not modify the simulator.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_h(directory, total_slots):
    samples = []
    for slot in range(total_slots):
        with np.load(directory / f"dmimochans_{slot}.npz") as data:
            samples.append(np.asarray(data["Hts"]))
    return np.stack(samples)


def dominant_modes(h, num_users=4):
    """Match the phase-1 experiment's two-slot/symbol-averaged CSI modes."""
    # h: [slot, rx_ant, tx_ant, ofdm_symbol, subcarrier]
    paired = 0.5 * (h[:-1, : 2 * num_users] + h[1:, : 2 * num_users])
    paired = paired.reshape(
        paired.shape[0], num_users, 2, paired.shape[2], paired.shape[3], paired.shape[4]
    ).mean(axis=4)
    # [slot, user, subcarrier, rx_ant, tx_ant]
    paired = np.transpose(paired, (0, 1, 4, 2, 3))
    _, singular_values, vh = np.linalg.svd(paired, full_matrices=False)
    return paired, singular_values[..., 0], vh[..., 0, :]


def grad_ascent_beams(modes, iterations=50, snr_db=20.0, correct_conjugation=False):
    """Calculate beams with the existing or mathematically consistent product."""
    v = modes / np.maximum(np.linalg.norm(modes, axis=-1, keepdims=True), 1e-12)
    snr = 10.0 ** (snr_db / 10.0)
    p = np.sum(np.sqrt(snr) * np.conj(v), axis=1)
    p /= np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)
    for _ in range(iterations):
        if correct_conjugation:
            # modes contains the channel row v^H returned by np.linalg.svd.
            inner = np.sum(v * p[:, None, :, :], axis=-1)
            gradient_direction = np.conj(v)
        else:
            # Reproduce np.vdot(vk, p) and vk*projection in the implementation.
            inner = np.sum(np.conj(v) * p[:, None, :, :], axis=-1)
            gradient_direction = v
        gamma = snr * np.abs(inner) ** 2
        weight = snr / (gamma + 1e-6)
        gradient = np.sum(
            weight[..., None] * gradient_direction * inner[..., None], axis=1
        )
        p = p + 0.2 * gradient
        p /= np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)
    return p


def delay_statistics(matrices, sigma1, modes, delays):
    stats = {}
    for delay in delays:
        current_h = matrices[delay:]
        stale_vh = modes[:-delay]
        current_vh = modes[delay:]

        alignment = np.abs(np.sum(current_vh * np.conj(stale_vh), axis=-1)) ** 2

        # np.linalg.svd returns v^H. The transmit beam is therefore conj(vh).
        stale_beam = np.conj(stale_vh)[..., None]
        stale_gain = np.sum(np.abs(current_h @ stale_beam) ** 2, axis=(-2, -1))
        oracle_gain = sigma1[delay:] ** 2
        gain_ratio = np.clip(stale_gain / np.maximum(oracle_gain, 1e-30), 1e-30, None)
        loss_db = -10.0 * np.log10(gain_ratio)

        stale_multicast_beam = grad_ascent_beams(stale_vh)
        oracle_multicast_beam = grad_ascent_beams(current_vh)
        stale_user_gain = np.mean(
            np.abs(np.einsum("tksra,tsa->tksr", current_h, stale_multicast_beam)) ** 2,
            axis=(-2, -1),
        )
        oracle_user_gain = np.mean(
            np.abs(np.einsum("tksra,tsa->tksr", current_h, oracle_multicast_beam)) ** 2,
            axis=(-2, -1),
        )
        # The common stream fails if the weakest user fails.
        stale_worst = np.min(stale_user_gain, axis=1)
        oracle_worst = np.min(oracle_user_gain, axis=1)
        multicast_loss_db = -10.0 * np.log10(
            np.maximum(stale_worst, 1e-30) / np.maximum(oracle_worst, 1e-30)
        )

        stale_fixed_beam = grad_ascent_beams(stale_vh, correct_conjugation=True)
        oracle_fixed_beam = grad_ascent_beams(current_vh, correct_conjugation=True)
        stale_fixed_gain = np.mean(
            np.abs(np.einsum("tksra,tsa->tksr", current_h, stale_fixed_beam)) ** 2,
            axis=(-2, -1),
        )
        oracle_fixed_gain = np.mean(
            np.abs(np.einsum("tksra,tsa->tksr", current_h, oracle_fixed_beam)) ** 2,
            axis=(-2, -1),
        )
        fixed_loss_db = -10.0 * np.log10(
            np.maximum(np.min(stale_fixed_gain, axis=1), 1e-30)
            / np.maximum(np.min(oracle_fixed_gain, axis=1), 1e-30)
        )

        stats[str(delay)] = {
            "samples": int(alignment.size),
            "dominant_mode_alignment_mean": float(np.mean(alignment)),
            "dominant_mode_alignment_p05": float(np.percentile(alignment, 5)),
            "single_user_beamforming_loss_db_mean": float(np.mean(loss_db)),
            "single_user_beamforming_loss_db_p95": float(np.percentile(loss_db, 95)),
            "multicast_worst_user_loss_db_mean": float(np.mean(multicast_loss_db)),
            "multicast_worst_user_loss_db_p95": float(np.percentile(multicast_loss_db, 95)),
            "corrected_multicast_worst_user_loss_db_mean": float(np.mean(fixed_loss_db)),
            "corrected_multicast_worst_user_loss_db_p95": float(np.percentile(fixed_loss_db, 95)),
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--total-slots", type=int, default=99)
    parser.add_argument("--rx-antenna", type=int, default=0)
    parser.add_argument("--tx-antenna", type=int, default=0)
    parser.add_argument("--ofdm-symbol", type=int, default=7)
    parser.add_argument("--subcarrier", type=int, default=256)
    parser.add_argument("--delays", nargs="+", type=int, default=[1, 4, 6, 8, 10])
    args = parser.parse_args()

    h = load_h(args.input, args.total_slots)
    coefficient = h[
        :, args.rx_antenna, args.tx_antenna, args.ofdm_symbol, args.subcarrier
    ]
    matrices, sigma1, modes = dominant_modes(h)
    stats = delay_statistics(matrices, sigma1, modes, args.delays)

    phase = np.unwrap(np.angle(coefficient))
    time_ms = np.arange(args.total_slots)

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.0), sharex=True)
    axes[0].plot(time_ms, coefficient.real, label="Real", lw=1.4)
    axes[0].plot(time_ms, coefficient.imag, label="Imaginary", lw=1.4)
    axes[0].set_ylabel("Coefficient")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_ms, np.abs(coefficient), color="tab:green", lw=1.5)
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(alpha=0.25)

    axes[2].plot(time_ms, phase, color="tab:purple", lw=1.5)
    axes[2].set_ylabel("Unwrapped phase (rad)")
    axes[2].set_xlabel("Time / slot (ms)")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        "highest_mobility, drop 1: TxSquad UE 1 Rx-ant 1, BS Tx-ant 1, "
        f"subcarrier {args.subcarrier}, OFDM symbol {args.ofdm_symbol}"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")

    result = {
        "selection": {
            "input": str(args.input),
            "rx_antenna": args.rx_antenna,
            "tx_antenna": args.tx_antenna,
            "ofdm_symbol": args.ofdm_symbol,
            "subcarrier": args.subcarrier,
            "slot_duration_ms": 1,
        },
        "coefficient": {
            "magnitude_min": float(np.min(np.abs(coefficient))),
            "magnitude_mean": float(np.mean(np.abs(coefficient))),
            "magnitude_max": float(np.max(np.abs(coefficient))),
            "unwrapped_phase_change_rad": float(phase[-1] - phase[0]),
        },
        "delay_statistics": stats,
    }
    args.metrics.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
