#!/usr/bin/env python3
"""Plot MU-MIMO throughput on test drops for multiple prediction baselines."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
    "savefig.dpi": 300,
})

STYLE = {
    "configured_wesn": {"label": "Configured WESN", "marker": "^", "color": "tab:green"},
    "two_mode": {"label": "Two-Mode WESN", "marker": "o", "color": "tab:blue"},
    "kalman_filter": {"label": "Kalman Filter", "marker": "s", "color": "tab:orange"},
    "channelmamba": {"label": "ChannelMamba", "marker": "D", "color": "tab:red"},
    "outdated_csi": {"label": "Outdated CSI", "marker": "x", "color": "0.45"},
}

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
        "channelmamba": [
            f"{prefix}_prediction_channelmamba_pmi_quantization_True.npz",
            f"{prefix}_prediction_channelmamba_pmi_quantization_True_imitation_none_steps_0.npz",
            f"{prefix}_prediction_channelmamba_pmi_quantization_True_drop_split.npz",
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
    p.add_argument("--mobility", default="highest_mobility")
    p.add_argument("--test-drops", default="11-20")
    p.add_argument("--rx-ues", type=int, default=4)
    p.add_argument("--tx-ues", default="2,4,6,8,10")
    p.add_argument("--output", default="results/figures/configured_wesn_drop_split_throughput.png")
    args = p.parse_args()

    lo, hi = [int(x) for x in args.test_drops.split("-")]
    drops = list(range(lo, hi + 1))
    tx_vals = [int(x) for x in args.tx_ues.split(",")]
    base = Path(args.base_dir)

    methods = ["configured_wesn", "two_mode", "kalman_filter", "channelmamba", "outdated_csi"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for method in methods:
        st = STYLE[method]
        y = []
        for tx in tx_vals:
            vals = [load_point(base, args.mobility, d, args.rx_ues, tx, method) for d in drops]
            vals = [v for v in vals if v is not None]
            y.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(
            np.asarray(tx_vals) + 1,
            y,
            marker=st["marker"],
            color=st["color"],
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=st["label"],
        )
    ax.set_xlabel("Number of RUs")
    ax.set_ylabel("Throughput (Mbps)")
    ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, length=5)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5)
    ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=9, handlelength=1.8, borderaxespad=0.2)
    fig.tight_layout(pad=0.2)
    out_base, _ = str(Path(args.output)).rsplit(".", 1)
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()