#!/usr/bin/env python3
"""Reproduce the structure-aware SS-KF/WESN counts in main.tex.

Counts are complex multiplications.  Online memory is reported for complex64
storage and includes all resource-element states for one UE--RU link.  The
count corresponds to identity activation; nonlinear activation latency must be
measured separately.
"""

from __future__ import annotations

import argparse
import math


def counts(d: int, p: int, r_state: int, b_re: int) -> dict[str, float]:
    z_dim = r_state + p * d

    # Fixed gain update K e followed by a companion-form AR prediction.
    kf_per_re = 2 * p * d * d

    # Dense input projection, elementwise diagonal recurrence, and readout.
    wesn_per_re = r_state * (p * d + 1) + d * z_dim

    # Shared coefficients plus all per-resource-element states/input windows.
    kf_memory = 2 * p * d * d + b_re * p * d
    wesn_memory = (
        r_state * p * d
        + r_state
        + d * z_dim
        + b_re * (r_state + p * d)
    )

    break_even_r = p * d * d / (d * (p + 1) + 1)
    return {
        "kf_per_re": kf_per_re,
        "wesn_per_re": wesn_per_re,
        "kf_per_slot": b_re * kf_per_re,
        "wesn_per_slot": b_re * wesn_per_re,
        "saving_percent": 100.0 * (kf_per_re - wesn_per_re) / kf_per_re,
        "kf_memory_mib": 8.0 * kf_memory / (1024.0**2),
        "wesn_memory_mib": 8.0 * wesn_memory / (1024.0**2),
        "break_even_r": break_even_r,
    }


def amortization_example(
    *,
    d: int,
    p: int,
    modes: int,
    poles: int,
    residue_rank: int,
    b_re: int,
    config_windows: int,
    ar_samples_per_re: int,
    training_samples_per_re: int,
    freq_samples: int,
    riccati_iters: int,
    vector_fit_iters: int,
    horizon: int,
) -> dict[str, float]:
    """Leading-term proxy used for the illustrative amortization table."""
    n_state = p * d
    r_state = modes * poles * residue_rank
    z_dim = r_state + n_state
    n_ar = ar_samples_per_re * b_re
    n_train = training_samples_per_re * b_re

    ar_cost = n_ar * n_state**2 + n_state**3
    # The KF pools the same samples that WESN divides among model windows.
    kf_ar_cost = config_windows * n_ar * n_state**2 + n_state**3
    kf_config = kf_ar_cost + riccati_iters * n_state**3

    transfer_per_window = freq_samples * (
        n_state**3 + n_state**2 * d
    )
    pca_cost = freq_samples * d**2 * config_windows**2
    # Four real/complex multiply-like operations per sparse LSQR nonzero pair.
    vector_fit_cost = (
        4
        * modes
        * vector_fit_iters
        * poles
        * freq_samples
        * d**2
    )
    residue_fit_cost = modes * (
        freq_samples * poles**2
        + freq_samples * poles * d**2
        + poles**3
    )
    low_rank_cost = modes * poles * d**2 * residue_rank + modes * poles
    wesn_config = (
        config_windows
        * (
            ar_cost
            + riccati_iters * n_state**3
            + transfer_per_window
        )
        + pca_cost
        + vector_fit_cost
        + residue_fit_cost
        + low_rank_cost
    )
    wesn_online_update = (
        n_train * r_state * (p * d + 1)
        + n_train * z_dim**2
        + n_train * d * z_dim
        + z_dim**3
    )
    inference = counts(d=d, p=p, r_state=r_state, b_re=b_re)
    kf_inference = inference["kf_per_slot"]
    wesn_inference = inference["wesn_per_slot"]
    saving_per_event = kf_inference - wesn_inference - wesn_online_update
    break_even = (
        (wesn_config - kf_config) / saving_per_event
        if saving_per_event > 0
        else math.inf
    )
    return {
        "kf_config": kf_config,
        "wesn_config": wesn_config,
        "wesn_online_update": wesn_online_update,
        "kf_inference": kf_inference,
        "wesn_inference": wesn_inference,
        "wesn_online_total": wesn_inference + wesn_online_update,
        "break_even": break_even,
        "kf_amortized": kf_inference + kf_config / horizon,
        "wesn_amortized": (
            wesn_inference + wesn_online_update + wesn_config / horizon
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar-order", type=int, default=2)
    parser.add_argument("--modes", type=int, default=1)
    parser.add_argument("--poles", type=int, default=2)
    parser.add_argument("--residue-rank", type=int, default=1)
    parser.add_argument("--subcarriers", type=int, default=512)
    parser.add_argument("--ofdm-symbols", type=int, default=14)
    parser.add_argument("--config-windows", type=int, default=16)
    parser.add_argument("--ar-samples-per-re", type=int, default=3)
    parser.add_argument("--training-samples-per-re", type=int, default=3)
    parser.add_argument("--freq-samples", type=int, default=64)
    parser.add_argument("--riccati-iters", type=int, default=50)
    parser.add_argument("--vector-fit-iters", type=int, default=50)
    parser.add_argument("--amortization-horizon", type=int, default=1000)
    parser.add_argument("--example-d", type=int, default=16)
    args = parser.parse_args()

    r_state = args.modes * args.poles * args.residue_rank
    b_re = args.subcarriers * args.ofdm_symbols
    print(
        f"P={args.ar_order}, M={args.modes}, K={args.poles}, "
        f"rank={args.residue_rank}, R={r_state}, B={b_re}"
    )
    print(
        "antennas D  KF/RE  WESN/RE  KF/slot       WESN/slot     "
        "saving[%]  KF MiB  WESN MiB  R threshold"
    )
    for n_rx, n_tx in ((2, 2), (4, 2), (4, 4), (8, 8)):
        d = n_rx * n_tx
        row = counts(d=d, p=args.ar_order, r_state=r_state, b_re=b_re)
        print(
            f"{n_rx}x{n_tx:<3} {d:2d} "
            f"{row['kf_per_re']:6.0f} {row['wesn_per_re']:8.0f} "
            f"{row['kf_per_slot']:12.0f} {row['wesn_per_slot']:13.0f} "
            f"{row['saving_percent']:9.2f} "
            f"{row['kf_memory_mib']:7.2f} {row['wesn_memory_mib']:9.2f} "
            f"{row['break_even_r']:11.2f}"
        )

    print("\nSystem sweep (five RX UEs)")
    print("TX UEs  links  KF/system      WESN/system    saving[%]")
    per_d = {
        d: counts(d=d, p=args.ar_order, r_state=r_state, b_re=b_re)
        for d in (4, 8, 16)
    }
    for num_tx_ues in (2, 4, 6, 8, 10):
        num_links = 6 * (num_tx_ues + 1)
        kf_system = (
            per_d[16]["kf_per_slot"]
            + (5 + num_tx_ues) * per_d[8]["kf_per_slot"]
            + 5 * num_tx_ues * per_d[4]["kf_per_slot"]
        )
        wesn_system = (
            per_d[16]["wesn_per_slot"]
            + (5 + num_tx_ues) * per_d[8]["wesn_per_slot"]
            + 5 * num_tx_ues * per_d[4]["wesn_per_slot"]
        )
        saving = 100.0 * (kf_system - wesn_system) / kf_system
        print(
            f"{num_tx_ues:6d} {num_links:6d} {kf_system:14.0f} "
            f"{wesn_system:14.0f} {saving:12.2f}"
        )

    example = amortization_example(
        d=args.example_d,
        p=args.ar_order,
        modes=args.modes,
        poles=args.poles,
        residue_rank=args.residue_rank,
        b_re=b_re,
        config_windows=args.config_windows,
        ar_samples_per_re=args.ar_samples_per_re,
        training_samples_per_re=args.training_samples_per_re,
        freq_samples=args.freq_samples,
        riccati_iters=args.riccati_iters,
        vector_fit_iters=args.vector_fit_iters,
        horizon=args.amortization_horizon,
    )
    print(f"\nAmortization proxy (D={args.example_d})")
    for key in (
        "kf_config",
        "wesn_config",
        "wesn_online_update",
        "kf_inference",
        "wesn_inference",
        "wesn_online_total",
        "kf_amortized",
        "wesn_amortized",
    ):
        print(f"{key:20s} {example[key]:.6e}")
    print(f"{'break_even_events':20s} {example['break_even']:.2f}")


if __name__ == "__main__":
    main()
