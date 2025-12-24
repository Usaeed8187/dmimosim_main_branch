"""Plot MU-MIMO KPI sweeps aggregated over multiple drops.

The plotting logic is tailored for the artifacts produced by
``sims/run_sim_mu_mimo_kpi_multiple_drops.sh``.  The script loads the
per-drop ``npz`` files, averages metrics across drops, and produces four
figures:

* Uncoded BER vs. number of Tx UEs (fixed Rx UEs, fixed MCS)
* Uncoded BER vs. number of Rx UEs (fixed Tx UEs, fixed MCS)
* Throughput vs. number of Tx UEs (fixed Rx UEs, best MCS per point)
* Throughput vs. number of Rx UEs (fixed Tx UEs, best MCS per point)

For BER plots, the modulation order and code rate are fixed by the
command-line arguments.  For throughput plots, the script selects, for
each data point, the MCS that maximizes the *average* throughput across
all requested drops and prints the maximizing MCS choices.
"""

from __future__ import annotations

import argparse
import glob

import os

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

################################################################################
# Argument parsing
################################################################################


def _float_or_fraction(value: str) -> float:
    """Parse a float or fraction string (e.g., ``"2/3"``).

    Args:
        value: String to parse.

    Returns:
        The parsed floating point value.
    """

    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return float(value)


@dataclass
class PlotConfig:
    base_dir: str
    mobility: str
    drops: Sequence[int]
    rx_ues: Sequence[int]
    tx_ues: Sequence[int]
    modulation_orders: Sequence[int]
    code_rates: Sequence[float]
    ber_modulation_order: int
    ber_code_rate: float
    fixed_rx_for_tx_sweep: int
    fixed_tx_for_rx_sweep: int
    output_dir: str
    prediction: bool = True


################################################################################
# Data loading helpers
################################################################################


@dataclass
class DataPoint:
    uncoded_ber: float
    throughput: float


class ResultLoader:
    def __init__(self, cfg: PlotConfig) -> None:
        self.cfg = cfg

    def _drop_folder(self, drop_id: int) -> str:
        folder_name = f"channels_{self.cfg.mobility}_{drop_id}"
        return os.path.join(self.cfg.base_dir, folder_name)

    def _suffix(self) -> str:
        return "_prediction" if self.cfg.prediction else ""

    @staticmethod
    def _parse_code_rate_from_path(path: str) -> Optional[float]:
        basename = os.path.basename(path)
        try:
            middle = basename.split("code_rate_")[1]
            code_rate_str = middle.split("_rx_UE")[0]
            return float(code_rate_str)
        except (IndexError, ValueError):
            return None

    def _find_file(
        self, drop_id: int, rx_ues: int, tx_ues: int, mod_order: int, code_rate: float
    ) -> Optional[str]:
        folder = self._drop_folder(drop_id)
        suffix = self._suffix()
        code_rate_str = str(code_rate)
        candidate = os.path.join(
            folder,
            f"mu_mimo_results_mod_order_{mod_order}_code_rate_{code_rate_str}_rx_UE_{rx_ues}_tx_UE_{tx_ues}{suffix}.npz",
        )
        if os.path.exists(candidate):
            return candidate

        pattern = os.path.join(
            folder,
            f"mu_mimo_results_mod_order_{mod_order}_code_rate_*_rx_UE_{rx_ues}_tx_UE_{tx_ues}{suffix}.npz",
        )
        matches = glob.glob(pattern)
        if not matches:
            return None

        target = float(code_rate)
        matches.sort(
            key=lambda path: abs(
                (self._parse_code_rate_from_path(path) or target) - target
            )
        )
        return matches[0]

    @staticmethod
    def _scalar_from_array(arr: np.ndarray) -> float:
        arr = np.asarray(arr)
        if arr.size == 0:
            return float("nan")
        return float(np.asarray(arr).reshape(-1)[-1])

    @staticmethod
    def _uncoded_ber_from_npz(data: np.lib.npyio.NpzFile) -> float:
        uncoded = data.get("uncoded_ber_list")
        if uncoded is None:
            return float("nan")
        uncoded_array = np.asarray(uncoded, dtype=float)
        return float(np.nanmean(uncoded_array))

    def load_datapoint(
        self, drop_id: int, rx_ues: int, tx_ues: int, mod_order: int, code_rate: float
    ) -> Optional[DataPoint]:
        file_path = self._find_file(drop_id, rx_ues, tx_ues, mod_order, code_rate)
        if file_path is None:
            return None

        with np.load(file_path, allow_pickle=True) as data:
            uncoded_ber = self._uncoded_ber_from_npz(data)
            throughput = self._scalar_from_array(np.atleast_1d(data.get("throughput", [])))
        return DataPoint(uncoded_ber=uncoded_ber, throughput=throughput)


################################################################################
# Aggregation
################################################################################


def aggregate_metrics(
    loader: ResultLoader,
    rx_values: Iterable[int],
    tx_values: Iterable[int],
    modulation_orders: Iterable[int],
    code_rates: Iterable[float],
) -> Dict[Tuple[int, int, int, float], List[DataPoint]]:
    results: Dict[Tuple[int, int, int, float], List[DataPoint]] = {}
    for drop_id in loader.cfg.drops:
        for rx_ues in rx_values:
            for tx_ues in tx_values:
                for mod_order in modulation_orders:
                    for code_rate in code_rates:
                        datapoint = loader.load_datapoint(
                            drop_id, rx_ues, tx_ues, mod_order, code_rate
                        )
                        if datapoint is None:
                            continue
                        results.setdefault(
                            (rx_ues, tx_ues, mod_order, float(code_rate)), []
                        ).append(datapoint)
    return results


def average_datapoints(points: Sequence[DataPoint]) -> DataPoint:
    return DataPoint(
        uncoded_ber=float(np.nanmean([p.uncoded_ber for p in points])),
        throughput=float(np.nanmean([p.throughput for p in points])),
    )


################################################################################
# Plotting helpers
################################################################################


def plot_metric(
    x_values: Sequence[int],
    y_values: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
) -> None:
    plt.figure()
    plt.plot(x_values, y_values, marker="o")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

def semilogy_metric(
    x_values: Sequence[int],
    y_values: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
) -> None:
    plt.figure()
    plt.semilogy(x_values, y_values, marker="o")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")



def select_best_mcs(
    aggregated: Dict[Tuple[int, int, int, float], List[DataPoint]],
    rx_ues: int,
    tx_ues: int,
    modulation_orders: Iterable[int],
    code_rates: Iterable[float],
) -> Tuple[Optional[float], Optional[Tuple[int, float]]]:
    best_throughput = None
    best_mcs: Optional[Tuple[int, float]] = None
    for mod_order in modulation_orders:
        for code_rate in code_rates:
            key = (rx_ues, tx_ues, mod_order, float(code_rate))
            if key not in aggregated:
                continue
            avg_point = average_datapoints(aggregated[key])
            if best_throughput is None or avg_point.throughput > best_throughput:
                best_throughput = avg_point.throughput
                best_mcs = (mod_order, float(code_rate))
    return best_throughput, best_mcs


################################################################################
# Main plotting routine
################################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=os.path.join("results", "channels_multiple_mu_mimo"),
        help="Root directory containing per-drop results.",
    )
    parser.add_argument("--mobility", default="high_mobility", help="Mobility string used in the folder names.")
    parser.add_argument(
        "--drops",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Drop indices to average over (e.g., 1 2 3).",
    )
    parser.add_argument(
        "--rx-ues",
        type=int,
        nargs="+",
        default=[0, 2, 4, 6],
        help="Rx UE counts that were simulated.",
    )
    parser.add_argument(
        "--tx-ues",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8, 10],
        help="Tx UE counts that were simulated (num_txue_sel).",
    )
    parser.add_argument(
        "--modulation-orders",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Modulation orders that were simulated (e.g., 2 for QPSK, 4 for 16-QAM).",
    )
    parser.add_argument(
        "--code-rates",
        type=_float_or_fraction,
        nargs="+",
        default=[_float_or_fraction("2/3"), _float_or_fraction("5/6")],
        help="Code rates that were simulated (accepts fractions like 2/3).",
    )
    parser.add_argument(
        "--ber-modulation-order",
        type=int,
        default=2,
        help="Modulation order to use for BER plots.",
    )
    parser.add_argument(
        "--ber-code-rate",
        type=_float_or_fraction,
        default=_float_or_fraction("2/3"),
        help="Code rate to use for BER plots (accepts fractions like 2/3).",
    )
    parser.add_argument(
        "--fixed-rx",
        type=int,
        default=4,
        help="Rx UE count to hold fixed when sweeping Tx UEs.",
    )
    parser.add_argument(
        "--fixed-tx",
        type=int,
        default=8,
        help="Tx UE count to hold fixed when sweeping Rx UEs.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("results", "plots"),
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--no-prediction",
        action="store_true",
        help="Look for non-prediction result files (omit the _prediction suffix).",
    )

    args = parser.parse_args()

    cfg = PlotConfig(
        base_dir=args.base_dir,
        mobility=args.mobility,
        drops=args.drops,
        rx_ues=args.rx_ues,
        tx_ues=args.tx_ues,
        modulation_orders=args.modulation_orders,
        code_rates=args.code_rates,
        ber_modulation_order=args.ber_modulation_order,
        ber_code_rate=args.ber_code_rate,
        fixed_rx_for_tx_sweep=args.fixed_rx,
        fixed_tx_for_rx_sweep=args.fixed_tx,
        output_dir=args.output_dir,
        prediction=not args.no_prediction,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    loader = ResultLoader(cfg)

    aggregated = aggregate_metrics(
        loader,
        cfg.rx_ues,
        cfg.tx_ues,
        cfg.modulation_orders,
        cfg.code_rates,
    )

    # BER vs Tx UEs (fixed Rx)
    ber_tx = []
    for tx in cfg.tx_ues:
        key = (cfg.fixed_rx_for_tx_sweep, tx, cfg.ber_modulation_order, float(cfg.ber_code_rate))
        points = aggregated.get(key, [])
        ber_tx.append(average_datapoints(points).uncoded_ber if points else np.nan)
    semilogy_metric(
        cfg.tx_ues,
        ber_tx,
        xlabel="Number of Tx UEs",
        ylabel="Uncoded BER",
        title=f"Uncoded BER vs Tx UEs (Rx UEs={cfg.fixed_rx_for_tx_sweep}, MCS={cfg.ber_modulation_order}/{cfg.ber_code_rate})",
        output_path=os.path.join(cfg.output_dir, "uncoded_ber_vs_tx_ues.png"),
    )

    # BER vs Rx UEs (fixed Tx)
    ber_rx = []
    for rx in cfg.rx_ues:
        key = (rx, cfg.fixed_tx_for_rx_sweep, cfg.ber_modulation_order, float(cfg.ber_code_rate))
        points = aggregated.get(key, [])
        ber_rx.append(average_datapoints(points).uncoded_ber if points else np.nan)
    semilogy_metric(
        cfg.rx_ues,
        ber_rx,
        xlabel="Number of Rx UEs",
        ylabel="Uncoded BER",
        title=f"Uncoded BER vs Rx UEs (Tx UEs={cfg.fixed_tx_for_rx_sweep}, MCS={cfg.ber_modulation_order}/{cfg.ber_code_rate})",
        output_path=os.path.join(cfg.output_dir, "uncoded_ber_vs_rx_ues.png"),
    )

    # Throughput vs Tx UEs (fixed Rx, best MCS)
    thr_tx = []
    best_mcs_tx = []
    for tx in cfg.tx_ues:
        best_throughput, best_mcs = select_best_mcs(
            aggregated,
            cfg.fixed_rx_for_tx_sweep,
            tx,
            cfg.modulation_orders,
            cfg.code_rates,
        )
        thr_tx.append(best_throughput if best_throughput is not None else np.nan)
        best_mcs_tx.append(best_mcs)
    plot_metric(
        cfg.tx_ues,
        thr_tx,
        xlabel="Number of Tx UEs",
        ylabel="Throughput",
        title=f"Throughput vs Tx UEs (Rx UEs={cfg.fixed_rx_for_tx_sweep}, best MCS)",
        output_path=os.path.join(cfg.output_dir, "throughput_vs_tx_ues.png"),
    )

    # Throughput vs Rx UEs (fixed Tx, best MCS)
    thr_rx = []
    best_mcs_rx = []
    for rx in cfg.rx_ues:
        best_throughput, best_mcs = select_best_mcs(
            aggregated,
            rx,
            cfg.fixed_tx_for_rx_sweep,
            cfg.modulation_orders,
            cfg.code_rates,
        )
        thr_rx.append(best_throughput if best_throughput is not None else np.nan)
        best_mcs_rx.append(best_mcs)
    plot_metric(
        cfg.rx_ues,
        thr_rx,
        xlabel="Number of Rx UEs",
        ylabel="Throughput",
        title=f"Throughput vs Rx UEs (Tx UEs={cfg.fixed_tx_for_rx_sweep}, best MCS)",
        output_path=os.path.join(cfg.output_dir, "throughput_vs_rx_ues.png"),
    )

    # Print the maximizing MCS selections for throughput plots
    print("\nMaximizing MCS for Throughput vs Tx UEs (Rx UEs fixed at {}):".format(cfg.fixed_rx_for_tx_sweep))
    for tx, mcs in zip(cfg.tx_ues, best_mcs_tx):
        print(f"  Tx UEs={tx}: {'None' if mcs is None else f'Mod {mcs[0]}, Code rate {mcs[1]}'})")

    print("\nMaximizing MCS for Throughput vs Rx UEs (Tx UEs fixed at {}):".format(cfg.fixed_tx_for_rx_sweep))
    for rx, mcs in zip(cfg.rx_ues, best_mcs_rx):
        print(f"  Rx UEs={rx}: {'None' if mcs is None else f'Mod {mcs[0]}, Code rate {mcs[1]}'})")


if __name__ == "__main__":
    main()