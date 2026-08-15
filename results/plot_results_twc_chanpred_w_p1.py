"""Compare perfect and imperfect phase-1 MU-MIMO results.

For every phase-2 channel-prediction method, this script overlays:

* ``perfect phase 1``: the phase-2-only pipeline, where every TxSquad node is
  given the original phase-2 payload without transmission errors;
* ``imperfect phase 1``: the phase-1-enabled pipeline, where each TxSquad node
  transmits the payload it decoded during phase 1.

The input layout, filename matching, KPI aggregation, and synchronization
sweeps match ``plot_results_twc_chanpred.py``. PDF, PNG, and SVG versions of
each figure are generated.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_results_twc_chanpred as phase2_plotter


METHODS = {
    # "configured_wesn": ("Configured WESN", "tab:green", "^"),
    "configured_wesn_balanced": ("Balanced Configured WESN", "tab:cyan", "X"),
    "configured_wesn_balanced_lite": (
        "Balanced Configured WESN-Lite",
        "tab:pink",
        "*",
    ),
    "wesn_lite": ("Low-Rank Configured WESN", "tab:brown", "v"),
    "kalman_filter": ("Kalman Filter", "tab:orange", "s"),
    "steady_state_kalman_filter": ("Steady-State KF", "tab:purple", "P"),
}


@dataclass(frozen=True)
class Phase1Scenario(phase2_plotter.Scenario):
    """A phase-2 prediction scenario plus its phase-1 reliability model."""

    imperfect_phase_1: bool = False

    @property
    def phase_1_label(self) -> str:
        return "imperfect phase 1" if self.imperfect_phase_1 else "perfect phase 1"

    @property
    def curve_label(self) -> str:
        return f"{self.label} ({self.phase_1_label})"


class Phase1ResultLoader(phase2_plotter.ResultLoader):
    """Load paired phase-2-only and phase-1-enabled result artifacts."""

    @staticmethod
    def _result_prefix(
        rx_ues: int,
        tx_ues: int,
        mod_order: int,
        code_rate: object,
        scenario: Phase1Scenario,
    ) -> str:
        if scenario.link_adapt:
            experiment = f"link_adapt_rx_UE_{rx_ues}_tx_UE_{tx_ues}"
        else:
            experiment = (
                f"mod_order_{mod_order}_code_rate_{code_rate}_"
                f"rx_UE_{rx_ues}_tx_UE_{tx_ues}"
            )
        if scenario.imperfect_phase_1:
            experiment = f"p1_True_p3_False_{experiment}"
        return f"mu_mimo_results_{experiment}"


def _comparison_scenarios(
    prediction_methods: Iterable[str], link_adapt: bool
) -> List[Phase1Scenario]:
    scenarios: List[Phase1Scenario] = []
    for method in prediction_methods:
        display_name = METHODS[method][0]
        for imperfect_phase_1 in (False, True):
            scenarios.append(
                Phase1Scenario(
                    perfect_csi=False,
                    prediction=True,
                    quantization=True,
                    label=display_name,
                    link_adapt=link_adapt,
                    prediction_method=method,
                    imperfect_phase_1=imperfect_phase_1,
                )
            )
    return scenarios


def _curve_style(scenario: Phase1Scenario) -> dict:
    _, color, marker = METHODS[scenario.prediction_method]
    return {
        "color": color,
        "marker": marker,
        "linestyle": "--" if scenario.imperfect_phase_1 else "-",
        "markerfacecolor": color if scenario.imperfect_phase_1 else "white",
    }


def _save_figure_multi_format(fig: plt.Figure, output_path: Path) -> None:
    base = output_path.with_suffix("")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {base}.pdf/.png/.svg")


def _plot_metric(
    x_values: Sequence[float],
    series: Sequence[Tuple[Phase1Scenario, Sequence[float]]],
    xlabel: str,
    ylabel: str,
    output_path: Path,
    logarithmic: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plot = ax.semilogy if logarithmic else ax.plot
    for scenario, y_values in series:
        style = _curve_style(scenario)
        plot(
            x_values,
            y_values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markerfacecolor=style["markerfacecolor"],
            markeredgewidth=1.2,
            linewidth=2.0,
            markersize=6.0,
            label=scenario.curve_label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(
        True,
        which="both" if logarithmic else "major",
        linestyle="-",
        linewidth=0.35,
        alpha=0.25,
    )
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, length=5)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5)
    ax.legend(
        frameon=False,
        loc="best",
        ncol=2,
        fontsize=8,
        handlelength=2.1,
    )
    fig.tight_layout(pad=0.3)
    _save_figure_multi_format(fig, output_path)


def _fixed_mcs_series(
    aggregated: Dict,
    scenarios: Sequence[Phase1Scenario],
    varying_values: Sequence[int],
    fixed_value: int,
    vary_tx: bool,
    mod_order: int,
    code_rate: float,
    metric: str,
) -> List[Tuple[Phase1Scenario, Sequence[float]]]:
    series = []
    for scenario in scenarios:
        values = []
        for varying_value in varying_values:
            rx_ues, tx_ues = (
                (fixed_value, varying_value)
                if vary_tx
                else (varying_value, fixed_value)
            )
            point = phase2_plotter._average_metric(
                aggregated,
                scenario,
                rx_ues,
                tx_ues,
                mod_order,
                float(code_rate),
            )
            values.append(getattr(point, metric) if point else np.nan)
        series.append((scenario, values))
    return series


def _throughput_series(
    aggregated: Dict,
    scenarios: Sequence[Phase1Scenario],
    varying_values: Sequence[int],
    fixed_value: int,
    vary_tx: bool,
    modulation_orders: Sequence[int],
    code_rates: Sequence[float],
) -> List[Tuple[Phase1Scenario, Sequence[float]]]:
    series = []
    for scenario in scenarios:
        values = []
        for varying_value in varying_values:
            rx_ues, tx_ues = (
                (fixed_value, varying_value)
                if vary_tx
                else (varying_value, fixed_value)
            )
            throughput, _ = phase2_plotter.select_best_mcs(
                aggregated,
                scenario,
                rx_ues,
                tx_ues,
                modulation_orders,
                code_rates,
            )
            values.append(throughput if throughput is not None else np.nan)
        series.append((scenario, values))
    return series


def _report_missing_pairs(
    loader: Phase1ResultLoader,
    scenarios: Sequence[Phase1Scenario],
    drops: Sequence[int],
    rx_ues: Sequence[int],
    tx_ues: Sequence[int],
    mod_order: int,
    code_rate: float,
) -> None:
    missing = {scenario: 0 for scenario in scenarios}
    for scenario in scenarios:
        for drop_id in drops:
            for rx in rx_ues:
                for tx in tx_ues:
                    if loader._find_file(
                        drop_id, rx, tx, mod_order, code_rate, scenario
                    ) is None:
                        missing[scenario] += 1
    for scenario, count in missing.items():
        if count:
            print(f"Missing {count} artifact(s): {scenario.curve_label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=SCRIPT_DIR / "channels_multiple_mu_mimo",
        help="Root directory containing channels_<mobility>_<drop> folders.",
    )
    parser.add_argument("--mobility", default="higher_mobility")
    parser.add_argument("--drops", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--rx-ues", type=int, nargs="+", default=[4])
    parser.add_argument("--tx-ues", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--modulation-orders", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--code-rates",
        type=phase2_plotter._float_or_fraction,
        nargs="+",
        default=[0.5],
    )
    parser.add_argument("--ber-modulation-order", type=int, default=4)
    parser.add_argument(
        "--ber-code-rate",
        type=phase2_plotter._float_or_fraction,
        default=0.5,
    )
    parser.add_argument("--fixed-rx", type=int, default=4)
    parser.add_argument("--fixed-tx", type=int, default=8)
    parser.add_argument(
        "--prediction-methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
    )
    parser.add_argument(
        "--wesn-lite-readout-mode",
        choices=["matched_ridge", "centered_ridge"],
        default="centered_ridge",
    )
    parser.add_argument(
        "--sync-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--sync-phase-std-deg", type=float, default=0.0)
    parser.add_argument("--sync-timing-std-samples", type=float, default=0.0)
    parser.add_argument(
        "--sync-phase-std-deg-values",
        type=float,
        nargs="+",
        default=[0.0, 3.6, 18.0, 36.0, 45.0, 90.0],
        help="Phase/frequency synchronization errors for the phase sweep.",
    )
    parser.add_argument(
        "--sync-timing-std-samples-values",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.5],
        help="Normalized timing-error standard deviations for the timing sweep.",
    )
    parser.add_argument(
        "--output-dir",
        default=SCRIPT_DIR / "plots_w_p1",
    )
    parser.add_argument(
        "--link-adapt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use link-adaptation artifacts (default: enabled).",
    )
    args = parser.parse_args()

    base_dir = phase2_plotter._resolve_path(str(args.base_dir), SCRIPT_DIR)
    output_dir = phase2_plotter._resolve_path(str(args.output_dir), SCRIPT_DIR)
    scenarios = _comparison_scenarios(args.prediction_methods, args.link_adapt)
    cfg = phase2_plotter.PlotConfig(
        base_dir=str(base_dir),
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
        output_dir=str(output_dir),
        link_adapt=args.link_adapt,
        scenarios=scenarios,
        channelmamba_seen_drops=[],
        channelmamba_all_drops=[],
        wesn_lite_readout_mode=args.wesn_lite_readout_mode,
        sync_errors=args.sync_errors,
        sync_phase_std_deg=args.sync_phase_std_deg,
        sync_timing_std_samples=args.sync_timing_std_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = Phase1ResultLoader(cfg)
    aggregated = phase2_plotter.aggregate_metrics(
        loader,
        scenarios,
        cfg.rx_ues,
        cfg.tx_ues,
        cfg.modulation_orders,
        cfg.code_rates,
    )

    tx_display = [tx + 1 for tx in cfg.tx_ues]
    rx_display = [rx + 2 for rx in cfg.rx_ues]
    plot_specs = [
        (
            "uncoded_ber",
            "Uncoded BER",
            "uncoded_ber_vs_tx_ues_w_p1.png",
        ),
        ("coded_ber", "Coded BER", "coded_ber_vs_tx_ues_w_p1.png"),
    ]
    for metric, ylabel, filename in plot_specs:
        _plot_metric(
            tx_display,
            _fixed_mcs_series(
                aggregated,
                scenarios,
                cfg.tx_ues,
                cfg.fixed_rx_for_tx_sweep,
                True,
                cfg.ber_modulation_order,
                cfg.ber_code_rate,
                metric,
            ),
            "Number of RUs",
            ylabel,
            output_dir / filename,
            logarithmic=True,
        )

    rx_plot_specs = [
        (
            "uncoded_ber",
            "Uncoded BER",
            "uncoded_ber_vs_rx_ues_w_p1.png",
        ),
        ("coded_ber", "Coded BER", "coded_ber_vs_rx_ues_w_p1.png"),
    ]
    for metric, ylabel, filename in rx_plot_specs:
        _plot_metric(
            rx_display,
            _fixed_mcs_series(
                aggregated,
                scenarios,
                cfg.rx_ues,
                cfg.fixed_tx_for_rx_sweep,
                False,
                cfg.ber_modulation_order,
                cfg.ber_code_rate,
                metric,
            ),
            "Number of UEs",
            ylabel,
            output_dir / filename,
            logarithmic=True,
        )

    _plot_metric(
        tx_display,
        _throughput_series(
            aggregated,
            scenarios,
            cfg.tx_ues,
            cfg.fixed_rx_for_tx_sweep,
            True,
            cfg.modulation_orders,
            cfg.code_rates,
        ),
        "Number of RUs",
        "Throughput (Mbps)",
        output_dir / "throughput_vs_tx_ues_w_p1.png",
    )
    _plot_metric(
        rx_display,
        _throughput_series(
            aggregated,
            scenarios,
            cfg.rx_ues,
            cfg.fixed_tx_for_rx_sweep,
            False,
            cfg.modulation_orders,
            cfg.code_rates,
        ),
        "Number of UEs",
        "Throughput (Mbps)",
        output_dir / "throughput_vs_rx_ues_w_p1.png",
    )

    _plot_metric(
        args.sync_timing_std_samples_values,
        phase2_plotter.sync_throughput_series(
            Phase1ResultLoader,
            cfg,
            scenarios,
            args.sync_timing_std_samples_values,
            sweep="timing",
        ),
        "Timing-error standard deviation (samples)",
        "Throughput (Mbps)",
        output_dir / "throughput_vs_sync_timing_std_samples_w_p1.png",
    )
    _plot_metric(
        args.sync_phase_std_deg_values,
        phase2_plotter.sync_throughput_series(
            Phase1ResultLoader,
            cfg,
            scenarios,
            args.sync_phase_std_deg_values,
            sweep="phase",
        ),
        "Phase-error standard deviation (degrees)",
        "Throughput (Mbps)",
        output_dir / "throughput_vs_sync_phase_std_deg_w_p1.png",
    )

    _report_missing_pairs(
        loader,
        scenarios,
        cfg.drops,
        cfg.rx_ues,
        cfg.tx_ues,
        cfg.ber_modulation_order,
        cfg.ber_code_rate,
    )


if __name__ == "__main__":
    main()
