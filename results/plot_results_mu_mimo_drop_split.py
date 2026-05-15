#!/usr/bin/env python3
"""Plot MU-MIMO throughput on test drops for multiple prediction baselines."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_point(base_dir: Path, mobility: str, drop: int, rx_ues: int, tx_ues: int, method: str) -> float | None:
    folder = base_dir / f"channels_{mobility}_{drop}"
    prefix = f"mu_mimo_results_link_adapt_rx_UE_{rx_ues}_tx_UE_{tx_ues}"

    candidates_by_method = {
        "configured_wesn": [
            f"{prefix}_prediction_configured_wesn_pmi_quantization_True_drop_split.npz",
        ],
        "two_mode": [
            f"{prefix}_prediction_two_mode_pmi_quantization_True.npz",
        ],
        "kalman_filter": [
            f"{prefix}_prediction_kalman_filter_pmi_quantization_True.npz",
        ],
        "outdated_csi": [
            f"{prefix}_perfect_CSI_False_pmi_quantization_True.npz",
        ],
    }
    path = None
    for pat in candidates_by_method.get(method, []):
        curr = folder / pat
        if curr.exists():
            path = curr
            break
    if path is None:
        return None
    data = np.load(path, allow_pickle=True)
    return float(np.asarray(data["throughput"]).reshape(-1)[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="results/channels_multiple_mu_mimo")
    p.add_argument("--mobility", default="higher_mobility")
    p.add_argument("--test-drops", default="11-20")
    p.add_argument("--rx-ues", type=int, default=4)
    p.add_argument("--tx-ues", default="2,4,6,8,10")
    p.add_argument("--output", default="results/figures/configured_wesn_drop_split_throughput.png")
    args = p.parse_args()

    lo, hi = [int(x) for x in args.test_drops.split("-")]
    drops = list(range(lo, hi + 1))
    tx_vals = [int(x) for x in args.tx_ues.split(",")]
    base = Path(args.base_dir)

    methods = [
        ("configured_wesn", "Configured WESN (drop split 50/50)", "o"),
        ("two_mode", "Two-Mode WESN", "s"),
        ("kalman_filter", "Kalman Filter", "^"),
        ("outdated_csi", "Outdated CSI", "d"),
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for method, label, marker in methods:
        y = []
        for tx in tx_vals:
            vals = [load_point(base, args.mobility, d, args.rx_ues, tx, method) for d in drops]
            vals = [v for v in vals if v is not None]
            y.append(float(np.mean(vals)) if vals else np.nan)
        plt.plot(tx_vals, y, marker=marker, label=label)
    plt.xlabel("Number of selected TX UEs")
    plt.ylabel("Throughput (Mbps)")
    plt.title("Throughput on test drops")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()