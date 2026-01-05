"""Plot DEQN MU-MIMO throughput and uncoded BER sweeps.

This utility targets the artifacts produced by
``sims/run_sim_mu_mimo_deqn_test_and_train.sh``.  It expects per-drop
results under ``results/channels_multiple_mu_mimo/channels_<mobility>_<drop>``
with file names of the form::

    mu_mimo_results_link_adapt_rx_UE_<rx>_tx_UE_<tx>_prediction_deqn_pmi_quantization_<quant>.npz
    mu_mimo_results_link_adapt_rx_UE_<rx>_tx_UE_<tx>_prediction_deqn_pmi_quantization_<quant>_imitation_<method>_steps_<steps>.npz

Throughput and uncoded BER values are averaged across drops for each
(rx UE, tx UE) pair.  The script produces two figures:

* Average throughput versus number of transmit UEs (one curve per RX-UE setting)
* Average uncoded BER versus number of transmit UEs (one curve per RX-UE setting)

Example usage::

    python results/plot_results_deqn_mu_mimo.py \
        --drops 1,2,3 \
        --rx-ues 0,2,4,6 \
        --tx-ues 2,4,6,8,10 \
        --output-dir results/plots_deqn
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PlotArgs:
    base_dir: str
    mobility: str
    drops: Sequence[int]
    rx_ues: Sequence[int]
    tx_ues: Sequence[int]
    quantization: bool
    imitation_method: str | None
    imitation_steps: int | None
    output_dir: str
    show: bool


@dataclass
class MetricPoint:
    throughput: List[float]
    uncoded_ber: List[float]


def _parse_int_list(csv: str) -> List[int]:
    return [int(item) for item in csv.split(",") if item]


def _build_filename(
    drop_dir: str,
    rx_ues: int,
    tx_ues: int,
    quantization: bool,
    imitation_method: str | None,
    imitation_steps: int | None,
) -> str:
    base = (
        "mu_mimo_results_link_adapt_rx_UE_"
        f"{rx_ues}_tx_UE_{tx_ues}_prediction_deqn_pmi_quantization_{quantization}"
    )
    if imitation_method and imitation_steps is not None:
        base += f"_imitation_{imitation_method}_steps_{imitation_steps}"
    return os.path.join(drop_dir, f"{base}.npz")


def _scalar_from_array(array: np.ndarray | float) -> float:
    """Convert stored arrays to scalar floats."""

    squeezed = np.asarray(array).squeeze()
    if squeezed.ndim == 0:
        return float(squeezed)
    return float(np.ravel(squeezed)[0])


def _load_metrics(args: PlotArgs) -> Dict[Tuple[int, int], MetricPoint]:
    metrics: Dict[Tuple[int, int], MetricPoint] = {}
    for drop in args.drops:
        drop_dir = os.path.join(
            args.base_dir, f"channels_{args.mobility}_{drop}"
        )
        if not os.path.isdir(drop_dir):
            print(f"Warning: drop directory not found: {drop_dir}")
            continue

        for rx in args.rx_ues:
            for tx in args.tx_ues:
                path = _build_filename(
                    drop_dir,
                    rx,
                    tx,
                    args.quantization,
                    args.imitation_method,
                    args.imitation_steps,
                )
                if not os.path.exists(path):
                    print(f"Warning: missing result file: {path}")
                    continue

                with np.load(path, allow_pickle=True) as data:
                    throughput_arr = data["throughput"]
                    ber_arr = data["uncoded_ber_list"]

                key = (rx, tx)
                point = metrics.setdefault(key, MetricPoint([], []))
                point.throughput.append(_scalar_from_array(throughput_arr))
                point.uncoded_ber.append(_scalar_from_array(ber_arr))

    return metrics


def _average_metrics(
    metrics: Dict[Tuple[int, int], MetricPoint]
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    averaged: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for key, point in metrics.items():
        if not point.throughput or not point.uncoded_ber:
            continue
        averaged[key] = (
            float(np.mean(point.throughput)),
            float(np.mean(point.uncoded_ber)),
        )
    return averaged


def _plot_curves(
    ax: plt.Axes,
    averaged: Dict[Tuple[int, int], Tuple[float, float]],
    rx_values: Sequence[int],
    tx_values: Sequence[int],
    metric_index: int,
    ylabel: str,
    title: str,
    yscale: str | None = None,
) -> None:
    for rx in rx_values:
        y_points: List[float] = []
        x_points: List[int] = []
        for tx in tx_values:
            key = (rx, tx)
            if key not in averaged:
                continue
            x_points.append(tx)
            y_points.append(averaged[key][metric_index])

        if not x_points:
            continue

        ax.plot(x_points, y_points, marker="o", label=f"RX UEs = {rx}")

    ax.set_xlabel("Number of TX UEs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    ax.grid(True)
    ax.legend()


def plot_results(args: PlotArgs) -> None:
    metrics = _load_metrics(args)
    averaged = _average_metrics(metrics)

    if not averaged:
        raise FileNotFoundError("No matching DEQN result files were found.")

    os.makedirs(args.output_dir, exist_ok=True)

    fig, (ax_thr, ax_ber) = plt.subplots(1, 2, figsize=(12, 5))
    _plot_curves(
        ax_thr,
        averaged,
        args.rx_ues,
        args.tx_ues,
        metric_index=0,
        ylabel="Average throughput (Mbps)",
        title="DEQN MU-MIMO Throughput vs. TX UEs",
    )
    _plot_curves(
        ax_ber,
        averaged,
        args.rx_ues,
        args.tx_ues,
        metric_index=1,
        ylabel="Average uncoded BER",
        title="DEQN MU-MIMO Uncoded BER vs. TX UEs",
        yscale="log",
    )

    plt.tight_layout()
    output_path = os.path.join(args.output_dir, "deqn_throughput_ber.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

def _parse_args() -> PlotArgs:
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_base_dir = os.path.join(script_dir, "channels_multiple_mu_mimo")
    default_output_dir = os.path.join(script_dir, "plots_deqn")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=default_base_dir,
        help="Root directory containing per-drop folders.",
    )
    parser.add_argument(
        "--mobility", default="high_mobility", help="Mobility tag used in folder names."
    )
    parser.add_argument(
        "--drops",
        default="1,2,3",
        help="Comma-separated list of drop identifiers to aggregate.",
    )
    parser.add_argument(
        "--rx-ues",
        default="0,2,4,6",
        help="Comma-separated list of RX-UE counts used in simulations.",
    )
    parser.add_argument(
        "--tx-ues",
        default="2,4,6,8,10",
        help="Comma-separated list of TX-UE counts used in simulations.",
    )
    parser.add_argument(
        "--quantization",
        default="True",
        help="Whether the filenames include quantization set to True/False.",
    )
    parser.add_argument(
        "--imitation-method",
        default=None,
        help="Optional imitation method tag used in filenames.",
    )
    parser.add_argument(
        "--imitation-steps",
        type=int,
        default=None,
        help="Optional imitation step count used in filenames.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display plots interactively instead of saving only.",
    )

    parsed = parser.parse_args()
    quantization = str(parsed.quantization).lower() in ("true", "1", "yes")
    drops = _parse_int_list(parsed.drops)
    rx_ues = _parse_int_list(parsed.rx_ues)
    tx_ues = _parse_int_list(parsed.tx_ues)

    return PlotArgs(
        base_dir=parsed.base_dir,
        mobility=parsed.mobility,
        drops=drops,
        rx_ues=rx_ues,
        tx_ues=tx_ues,
        quantization=quantization,
        imitation_method=parsed.imitation_method,
        imitation_steps=parsed.imitation_steps,
        output_dir=parsed.output_dir,
        show=parsed.show,
    )


if __name__ == "__main__":
    plot_results(_parse_args())