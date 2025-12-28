"""Plot MU-MIMO KPI sweeps aggregated over multiple drops.

The plotting logic is tailored for the artifacts produced by
``sims/run_sim_mu_mimo_kpi_multiple_drops.sh``.  The script loads the
per-drop ``npz`` files, averages metrics across drops, and produces four
figures:

* Uncoded BER vs. number of RUs (fixed UEs, fixed MCS)
* Uncoded BER vs. number of UEs (fixed RUs, fixed MCS)
* Throughput vs. number of RUs (fixed UEs, best MCS per point)
* Throughput vs. number of UEs (fixed RUs, best MCS per point)

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
    scenarios: Sequence["Scenario"]


################################################################################
# Data loading helpers
################################################################################


@dataclass
class DataPoint:
    uncoded_ber: float
    throughput: float

@dataclass(frozen=True)
class Scenario:
    perfect_csi: bool
    prediction: bool
    quantization: bool
    label: str
    prediction_method: Optional[str] = None


class ResultLoader:
    def __init__(self, cfg: PlotConfig) -> None:
        self.cfg = cfg

    def _drop_folder(self, drop_id: int) -> str:
        folder_name = f"channels_{self.cfg.mobility}_{drop_id}"
        return os.path.join(self.cfg.base_dir, folder_name)
    
    @staticmethod
    def _parse_code_rate_from_path(path: str) -> Optional[float]:
        basename = os.path.basename(path)
        try:
            middle = basename.split("code_rate_")[1]
            code_rate_str = middle.split("_rx_UE")[0]
            return float(code_rate_str)
        except (IndexError, ValueError):
            return None
        
    def _prediction_patterns(
        self,
        prefix: str,
        scenario: Scenario,
    ) -> List[str]:
        quant_str = str(scenario.quantization)
        method = scenario.prediction_method
        if method:
            return [
                f"{prefix}_prediction_{method}_pmi_quantization_{quant_str}.npz",
                f"{prefix}_prediction_{method}.npz",
            ]

        return [
            f"{prefix}_prediction_pmi_quantization_{quant_str}.npz",
            # Backward compatibility with the legacy naming (no method/quantization).
            f"{prefix}_prediction.npz",
        ]

    def _non_prediction_patterns(
        self,
        prefix: str,
        scenario: Scenario,
    ) -> List[str]:
        perfect_str = str(scenario.perfect_csi)
        quant_str = str(scenario.quantization)
        return [
            f"{prefix}_perfect_CSI_{perfect_str}_pmi_quantization_{quant_str}.npz",
            # Backward compatibility with the older perfect CSI naming.
            f"{prefix}_perfect_CSI_{perfect_str}.npz",
        ]

    def _candidate_paths(
        self,
        folder: str,
        prefix: str,
        scenario: Scenario,
    ) -> List[str]:
        patterns = (
            self._prediction_patterns(prefix, scenario)
            if scenario.prediction
            else self._non_prediction_patterns(prefix, scenario)
        )

        candidates: List[str] = []
        for pattern in patterns:
            full_pattern = os.path.join(folder, pattern)
            if "*" in pattern:
                matches = glob.glob(full_pattern)
                matches.sort()
                candidates.extend(matches)
            else:
                candidates.append(full_pattern)
        return candidates

    def _find_file(
        self,
        drop_id: int,
        rx_ues: int,
        tx_ues: int,
        mod_order: int,
        code_rate: float,
        scenario: Scenario,
    ) -> Optional[str]:
        folder = self._drop_folder(drop_id)
        code_rate_str = str(code_rate)
        prefix = (
            f"mu_mimo_results_mod_order_{mod_order}_code_rate_{code_rate_str}_rx_UE_{rx_ues}_tx_UE_{tx_ues}"
        )
        for candidate in self._candidate_paths(folder, prefix, scenario):
            if os.path.exists(candidate):
                return candidate

        # As a fallback, try to match slightly different code-rate strings.
        pattern = os.path.join(
            folder,
            f"mu_mimo_results_mod_order_{mod_order}_code_rate_*_rx_UE_{rx_ues}_tx_UE_{tx_ues}*.npz",
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
        for match in matches:
            if scenario.prediction and "prediction" in os.path.basename(match):
                return match
            if not scenario.prediction and "perfect_CSI" in os.path.basename(match):
                return match
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
        self,
        drop_id: int,
        rx_ues: int,
        tx_ues: int,
        mod_order: int,
        code_rate: float,
        scenario: Scenario,
    ) -> Optional[DataPoint]:
        file_path = self._find_file(
            drop_id, rx_ues, tx_ues, mod_order, code_rate, scenario
        )
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
    scenarios: Iterable[Scenario],
    rx_values: Iterable[int],
    tx_values: Iterable[int],
    modulation_orders: Iterable[int],
    code_rates: Iterable[float],
) -> Dict[Tuple[Scenario, int, int, int, float], List[DataPoint]]:
    results: Dict[Tuple[Scenario, int, int, int, float], List[DataPoint]] = {}
    for scenario in scenarios:
        for drop_id in loader.cfg.drops:
            for rx_ues in rx_values:
                for tx_ues in tx_values:
                    for mod_order in modulation_orders:
                        for code_rate in code_rates:
                            datapoint = loader.load_datapoint(
                                drop_id, rx_ues, tx_ues, mod_order, code_rate, scenario
                            )
                            if datapoint is None:
                                continue
                            results.setdefault(
                                (scenario, rx_ues, tx_ues, mod_order, float(code_rate)), []
                            ).append(datapoint)
    return results

def _average_metric(
    aggregated: Dict[Tuple[Scenario, int, int, int, float], List[DataPoint]],
    scenario: Scenario,
    rx_ues: int,
    tx_ues: int,
    mod_order: int,
    code_rate: float,
) -> Optional[DataPoint]:
    key = (scenario, rx_ues, tx_ues, mod_order, float(code_rate))
    points = aggregated.get(key, [])
    return average_datapoints(points) if points else None

def average_datapoints(points: Sequence[DataPoint]) -> DataPoint:
    return DataPoint(
        uncoded_ber=float(np.nanmean([p.uncoded_ber for p in points])),
        throughput=float(np.nanmean([p.throughput for p in points])),
    )

def _default_scenarios(include_prediction: bool = True) -> List[Scenario]:
    scenarios = [
        Scenario(
            perfect_csi=False,
            prediction=False,
            quantization=True,
            label="Worst case: Outdated CSI",
        ),
        Scenario(
            perfect_csi=False,
            prediction=True,
            quantization=True,
            label="Two-Mode WESN prediction",
            prediction_method=None,
        ),
        Scenario(
            perfect_csi=False,
            prediction=True,
            quantization=True,
            label="Wiener filter prediction",
            prediction_method="weiner_filter",
        ),
        Scenario(
            perfect_csi=True,
            prediction=False,
            quantization=False,
            label="Perfect CSI at BS (no quantization)",
        ),
        Scenario(
            perfect_csi=True,
            prediction=False,
            quantization=True,
            label="Perfect channel estimation, quantized feedback",
        ),
    ]

    if include_prediction:
        return scenarios
    return [scenario for scenario in scenarios if not scenario.prediction]

################################################################################
# Plotting helpers
################################################################################


def plot_metric(
    x_values: Sequence[int],
    series: Sequence[Tuple[str, Sequence[float]]],    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
) -> None:
    plt.figure()
    for label, y_values in series:
        plt.plot(x_values, y_values, marker="o", label=label)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

def semilogy_metric(
    x_values: Sequence[int],
    series: Sequence[Tuple[str, Sequence[float]]],    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
) -> None:
    plt.figure()
    for label, y_values in series:
        plt.semilogy(x_values, y_values, marker="o", label=label)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")



def select_best_mcs(
    aggregated: Dict[Tuple[Scenario, int, int, int, float], List[DataPoint]],
    scenario: Scenario,
    rx_ues: int,
    tx_ues: int,
    modulation_orders: Iterable[int],
    code_rates: Iterable[float],
) -> Tuple[Optional[float], Optional[Tuple[int, float]]]:
    best_throughput = None
    best_mcs: Optional[Tuple[int, float]] = None
    for mod_order in modulation_orders:
        for code_rate in code_rates:
            key = (scenario, rx_ues, tx_ues, mod_order, float(code_rate))
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
        default=[1, 2],
        help="Drop indices to average over (e.g., 1 2 3).",
    )
    parser.add_argument(
        "--rx-ues",
        type=int,
        nargs="+",
        default=[0, 2, 4, 6],
        help="UE counts that were simulated.",
    )
    parser.add_argument(
        "--tx-ues",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8, 10],
        help="RU counts that were simulated (num_txue_sel).",
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
        default=4,
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
        help="UE count to hold fixed when sweeping RUs.",
    )
    parser.add_argument(
        "--fixed-tx",
        type=int,
        default=8,
        help="RU count to hold fixed when sweeping UEs.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("results", "plots"),
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--no-prediction",
        action="store_true",
        help="Exclude CSI prediction curves from the plots.",
    )

    args = parser.parse_args()

    scenarios = _default_scenarios(include_prediction=not args.no_prediction)

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
        scenarios=scenarios,
        )

    os.makedirs(cfg.output_dir, exist_ok=True)
    loader = ResultLoader(cfg)

    aggregated = aggregate_metrics(
        loader,
        cfg.scenarios,
        cfg.rx_ues,
        cfg.tx_ues,
        cfg.modulation_orders,
        cfg.code_rates,
    )

    # BER vs RUs (fixed Rx)
    ber_tx_series = []
    for scenario in cfg.scenarios:
        scenario_values = []
        for tx in cfg.tx_ues:
            datapoint = _average_metric(
                aggregated,
                scenario,
                cfg.fixed_rx_for_tx_sweep,
                tx,
                cfg.ber_modulation_order,
                float(cfg.ber_code_rate),
            )
            scenario_values.append(datapoint.uncoded_ber if datapoint else np.nan)
        ber_tx_series.append((scenario.label, scenario_values))
    
    semilogy_metric(
        cfg.tx_ues,
        ber_tx_series,
        xlabel="Number of RUs",
        ylabel="Uncoded BER",
        title=f"Uncoded BER vs RUs (UEs={cfg.fixed_rx_for_tx_sweep}, MCS={cfg.ber_modulation_order}/{cfg.ber_code_rate})",
        output_path=os.path.join(cfg.output_dir, "uncoded_ber_vs_tx_ues.png"),
    )

    # BER vs UEs (fixed Tx)
    ber_rx_series = []
    for scenario in cfg.scenarios:
        scenario_values = []
        for rx in cfg.rx_ues:
            datapoint = _average_metric(
                aggregated,
                scenario,
                rx,
                cfg.fixed_tx_for_rx_sweep,
                cfg.ber_modulation_order,
                float(cfg.ber_code_rate),
            )
            scenario_values.append(datapoint.uncoded_ber if datapoint else np.nan)
        ber_rx_series.append((scenario.label, scenario_values))
    semilogy_metric(
        cfg.rx_ues,
        ber_rx_series,
        xlabel="Number of UEs",
        ylabel="Uncoded BER",
        title=f"Uncoded BER vs UEs (RUs={cfg.fixed_tx_for_rx_sweep}, MCS={cfg.ber_modulation_order}/{cfg.ber_code_rate})",
        output_path=os.path.join(cfg.output_dir, "uncoded_ber_vs_rx_ues.png"),
    )

    # Throughput vs RUs (fixed Rx, best MCS)
    thr_tx_series = []
    best_mcs_tx = {}
    for scenario in cfg.scenarios:
        scenario_thr = []
        scenario_best_mcs = []
        for tx in cfg.tx_ues:
            best_throughput, best_mcs = select_best_mcs(
                aggregated,
                scenario,
                cfg.fixed_rx_for_tx_sweep,
                tx,
                cfg.modulation_orders,
                cfg.code_rates,
            )
            scenario_thr.append(
                best_throughput if best_throughput is not None else np.nan
            )
            scenario_best_mcs.append(best_mcs)
        thr_tx_series.append((scenario.label, scenario_thr))
        best_mcs_tx[scenario] = scenario_best_mcs
    
    plot_metric(
        cfg.tx_ues,
        thr_tx_series,
        xlabel="Number of RUs",
        ylabel="Throughput",
        title=f"Throughput vs RUs (UEs={cfg.fixed_rx_for_tx_sweep}, best MCS)",
        output_path=os.path.join(cfg.output_dir, "throughput_vs_tx_ues.png"),
    )

    # Throughput vs UEs (fixed Tx, best MCS)
    thr_rx_series = []
    best_mcs_rx = {}
    for scenario in cfg.scenarios:
        scenario_thr = []
        scenario_best_mcs = []
        for rx in cfg.rx_ues:
            best_throughput, best_mcs = select_best_mcs(
                aggregated,
                scenario,
                rx,
                cfg.fixed_tx_for_rx_sweep,
                cfg.modulation_orders,
                cfg.code_rates,
            )
            scenario_thr.append(
                best_throughput if best_throughput is not None else np.nan
            )
            scenario_best_mcs.append(best_mcs)
        thr_rx_series.append((scenario.label, scenario_thr))
        best_mcs_rx[scenario] = scenario_best_mcs
    
    plot_metric(
        cfg.rx_ues,
        thr_rx_series,
        xlabel="Number of UEs",
        ylabel="Throughput",
        title=f"Throughput vs UEs (RUs={cfg.fixed_tx_for_rx_sweep}, best MCS)",
        output_path=os.path.join(cfg.output_dir, "throughput_vs_rx_ues.png"),
    )

    # Print the maximizing MCS selections for throughput plots
    print("\nMaximizing MCS for Throughput vs RUs (UEs fixed at {}):".format(cfg.fixed_rx_for_tx_sweep))
    for scenario in cfg.scenarios:
        print(f"  Scenario: {scenario.label}")
        for tx, mcs in zip(cfg.tx_ues, best_mcs_tx.get(scenario, [])):
            print(
                f"    RUs={tx}: {'None' if mcs is None else f'Mod {mcs[0]}, Code rate {mcs[1]}'}"
            )
    print("\nMaximizing MCS for Throughput vs UEs (RUs fixed at {}):".format(cfg.fixed_tx_for_rx_sweep))
    for scenario in cfg.scenarios:
        print(f"  Scenario: {scenario.label}")
        for rx, mcs in zip(cfg.rx_ues, best_mcs_rx.get(scenario, [])):
            print(
                f"    UEs={rx}: {'None' if mcs is None else f'Mod {mcs[0]}, Code rate {mcs[1]}'}"
            )

if __name__ == "__main__":
    main()