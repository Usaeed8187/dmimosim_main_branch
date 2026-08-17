"""Plot channel-prediction NMSE CDFs for MU-MIMO experiments.

This script loads ``chan_pred_nmse`` arrays saved in MU-MIMO result ``.npz`` files
and compares the channel predictors used by ``plot_results_twc_chanpred.py``.
This includes Two-Mode WESN, full and steady-state Kalman filters, configured
WESN, balanced configured WESN, low-rank configured WESN (WESN-Lite), and
ChannelMamba.

By default it searches under ``results/channels_multiple_mu_mimo`` and expects
subfolders named ``channels_<mobility>_<drop>``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

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

@dataclass(frozen=True)
class Scenario:
    label: str
    prediction: bool
    prediction_method: Optional[str] = None


def _float_or_fraction(value: str) -> float:
    """Parse a float or fraction string (e.g., ``2/3``)."""

    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return float(value)


def _resolve_path(path: str, relative_to: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (relative_to / candidate).resolve()


def _build_filename_prefix(
    link_adapt: bool,
    mod_order: int,
    code_rate: float,
    rx_ues: int,
    tx_ues: int,
) -> str:
    if link_adapt:
        return f"mu_mimo_results_link_adapt_rx_UE_{rx_ues}_tx_UE_{tx_ues}"
    return (
        f"mu_mimo_results_mod_order_{mod_order}_code_rate_{code_rate}"
        f"_rx_UE_{rx_ues}_tx_UE_{tx_ues}"
    )


def _candidate_paths(
    drop_folder: Path,
    prefix: str,
    quantization: bool,
    scenario: Scenario,
    sync_errors: bool = False,
    sync_frequency_std_ppb: float = 0.0,
    sync_initial_timing_std_ps: float = 0.0,
    sync_initial_phase_std_deg: float = 0.0,
    sync_phase_noise_s100_dbchz: Optional[float] = None,
    feedback_delay_ms: float = 4.0,
) -> List[Path]:
    quant_str = str(quantization)
    if scenario.prediction:
        assert scenario.prediction_method is not None
        patterns = [
            f"{prefix}_prediction_{scenario.prediction_method}_pmi_quantization_{quant_str}.npz",
            (
                f"{prefix}_prediction_{scenario.prediction_method}_"
                f"pmi_quantization_{quant_str}_imitation_none_steps_0.npz"
            ),
        ]

        if scenario.prediction_method in (
            "configured_wesn",
            "configured_wesn_balanced",
            "configured_wesn_balanced_lite",
            "wesn_lite",
        ):
            suffixes = [
                "_time_split_readout_*_workers_*",
                "_time_split_readout_*",
                "_time_split_workers_*",
                "_time_split",
                "",
            ]
        elif scenario.prediction_method == "channelmamba":
            suffixes = ["_time_split_workers_*", "_time_split"]
        else:
            # Match plot_results_twc_chanpred.py: current predictor runs encode
            # their inference worker count in the filename, while legacy runs
            # are unsuffixed.
            suffixes = ["_workers_*", ""]
    else:
        patterns = [
            f"{prefix}_perfect_CSI_False_pmi_quantization_{quant_str}.npz",
            f"{prefix}_perfect_CSI_False.npz",
        ]
        suffixes = [""]

    candidates: List[Path] = []
    token = lambda value: format(float(value), "g").replace("-", "m").replace(".", "p")
    pn_token = (
        "off"
        if sync_phase_noise_s100_dbchz is None
        else token(sync_phase_noise_s100_dbchz)
    )
    sync_suffix = (
        "_sync_clock_v2"
        f"_freq_std_ppb_{token(sync_frequency_std_ppb)}"
        f"_timing0_std_ps_{token(sync_initial_timing_std_ps)}"
        f"_phase0_std_deg_{token(sync_initial_phase_std_deg)}"
        f"_pn_s100_dbchz_{pn_token}"
    )
    feedback_suffix = f"_fb_delay_ms_{token(feedback_delay_ms)}"
    for pattern in patterns:
        stem = pattern[:-4] if pattern.endswith(".npz") else pattern
        for suffix in suffixes:
            candidates.extend(
                sorted(
                    drop_folder.glob(
                        f"{stem}{suffix}{sync_suffix}{feedback_suffix}.npz"
                    )
                )
            )
            if np.isclose(feedback_delay_ms, 4.0):
                candidates.extend(
                    sorted(drop_folder.glob(f"{stem}{suffix}{sync_suffix}.npz"))
                )
    if (
        not candidates
        and not sync_errors
        and sync_frequency_std_ppb == 0
        and sync_initial_timing_std_ps == 0
        and sync_initial_phase_std_deg == 0
        and sync_phase_noise_s100_dbchz is None
    ):
        for pattern in patterns:
            stem = pattern[:-4] if pattern.endswith(".npz") else pattern
            for suffix in suffixes:
                candidates.extend(
                    path
                    for path in sorted(drop_folder.glob(f"{stem}{suffix}.npz"))
                    if "_sync_clock_" not in path.name
                )
    return candidates


def _wesn_lite_readout_mode(path: Path) -> Optional[str]:
    """Read the WESN-Lite algorithm mode from explicit or legacy metadata."""

    try:
        with np.load(path, allow_pickle=True) as data:
            explicit = data.get("wesn_lite_readout_mode")
            if explicit is not None:
                return str(np.asarray(explicit).reshape(-1)[0]).replace("-", "_")

            raw = data.get("predictor_complexity_raw_json")
            if raw is None:
                return None
            metrics = json.loads(str(np.asarray(raw).item()))
            modes = {
                str(value.get("readout_objective", "")).replace("-", "_")
                for value in metrics.get("per_link_predictor_phases", {}).values()
                if value.get("readout_objective")
            }
            return next(iter(modes)) if len(modes) == 1 else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _choose_candidate(
    candidates: Sequence[Path],
    scenario: Scenario,
    wesn_lite_readout_mode: str,
) -> Optional[Path]:
    if not candidates:
        return None

    if scenario.prediction_method == "wesn_lite":
        matching = [
            path
            for path in candidates
            if _wesn_lite_readout_mode(path) == wesn_lite_readout_mode
        ]
        return (
            max(matching, key=lambda path: path.stat().st_mtime)
            if matching
            else None
        )

    # Keep artifact selection consistent with the throughput plot. In
    # particular, do not silently pair NMSE from a legacy unsuffixed KF run
    # with throughput from a newer ``_workers_*`` run.
    if scenario.prediction_method != "configured_wesn":
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return candidates[0]


def _load_nmse_from_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "chan_pred_nmse" not in data:
            raise KeyError(f"Missing key 'chan_pred_nmse' in {path}")
        nmse = np.asarray(data["chan_pred_nmse"]).astype(float).reshape(-1)

    nmse = nmse[np.isfinite(nmse)]
    return nmse


def _collect_nmse(
    base_dir: Path,
    mobility: str,
    drops: Sequence[int],
    prefixes: Sequence[str],
    quantization: bool,
    scenarios: Sequence[Scenario],
    wesn_lite_readout_mode: str,
    sync_errors: bool = False,
    sync_frequency_std_ppb: float = 0.0,
    sync_initial_timing_std_ps: float = 0.0,
    sync_initial_phase_std_deg: float = 0.0,
    sync_phase_noise_s100_dbchz: Optional[float] = None,
    feedback_delay_ms: float = 4.0,
) -> Dict[str, np.ndarray]:
    values_by_label: Dict[str, List[np.ndarray]] = {s.label: [] for s in scenarios}

    for drop in drops:
        drop_folder = base_dir / f"channels_{mobility}_{drop}"
        for prefix in prefixes:
            for scenario in scenarios:
                candidates = _candidate_paths(
                    drop_folder,
                    prefix,
                    quantization,
                    scenario,
                    sync_errors,
                    sync_frequency_std_ppb,
                    sync_initial_timing_std_ps,
                    sync_initial_phase_std_deg,
                    sync_phase_noise_s100_dbchz,
                    feedback_delay_ms,
                )
                chosen = _choose_candidate(
                    candidates, scenario, wesn_lite_readout_mode
                )
                if chosen is None:
                    print(
                        f"[WARN] Missing file for scenario={scenario.label}, drop={drop}, "
                        f"prefix={prefix}, folder={drop_folder}"
                    )
                    continue

                try:
                    nmse = _load_nmse_from_npz(chosen)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Failed loading {chosen}: {exc}")
                    continue

                if nmse.size == 0:
                    print(
                        f"[WARN] Empty chan_pred_nmse in scenario={scenario.label}, "
                        f"drop={drop}, file={chosen}"
                    )
                    continue

                values_by_label[scenario.label].append(nmse)

    merged: Dict[str, np.ndarray] = {}
    for label, chunks in values_by_label.items():
        merged[label] = np.concatenate(chunks) if chunks else np.array([], dtype=float)
    return merged

def _remove_outliers_by_percentile(
    values: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> np.ndarray:
    if values.size == 0:
        return values

    low = np.percentile(values, lower_percentile)
    high = np.percentile(values, upper_percentile)
    return values[(values >= low) & (values <= high)]


def _apply_outlier_filter(
    nmse_values: Dict[str, np.ndarray],
    lower_percentile: float,
    upper_percentile: float,
) -> Dict[str, np.ndarray]:
    if lower_percentile <= 0.0 and upper_percentile >= 100.0:
        return nmse_values

    filtered: Dict[str, np.ndarray] = {}
    for label, values in nmse_values.items():
        updated = _remove_outliers_by_percentile(values, lower_percentile, upper_percentile)
        print(
            f"[INFO] Outlier filtering ({lower_percentile:.2f}-{upper_percentile:.2f} pct), "
            f"{label}: kept {updated.size}/{values.size} samples"
        )
        filtered[label] = updated
    return filtered

def _save_cdf_figure(fig, output_path: Path, name: str) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    base = output_path / name
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved figure to: {base.with_suffix('.pdf')}")


def _parse_combined_label(combined_label: str) -> tuple[str, str]:
    if " (" in combined_label and combined_label.endswith(")"):
        method_label, mobility = combined_label.rsplit(" (", 1)
        mobility = mobility[:-1]
    else:
        method_label = combined_label
        mobility = "high_mobility"
    return method_label, mobility


def _plot_cdf(
    nmse_values: Dict[str, np.ndarray],
    nmse_in_db: bool,
    title: str,
    output_path: Path,
) -> None:
    method_colors = {
        "Two-mode WESN": "tab:blue",
        "Two-Mode WESN": "tab:blue",
        "Configured WESN": "tab:green",
        "Balanced Configured WESN": "tab:cyan",
        "Balanced Configured WESN-Lite": "tab:pink",
        "Low-Rank Configured WESN": "tab:brown",
        "Kalman Filter": "tab:orange",
        "Steady-State KF": "tab:purple",
        "ChannelMamba": "tab:red",
    }

    mobility_display = {
        "high_mobility": "10 km/h",
        "higher_mobility": "40 km/h",
        "highest_mobility": "80 km/h",
    }

    mobility_styles = {
        "high_mobility": ("-", "10 km/h"),
        "higher_mobility": ("--", "40 km/h"),
        "highest_mobility": (":", "80 km/h"),
    }

    xlabel = "Channel prediction NMSE (dB)" if nmse_in_db else "Channel prediction NMSE"

    # ------------------------------------------------------------------
    # Version 1: Original ellipse-style figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    plotted_methods = set()

    for combined_label, values in nmse_values.items():
        if values.size == 0:
            continue

        method_label, mobility = _parse_combined_label(combined_label)
        color = method_colors.get(method_label, None)

        to_plot = 10.0 * np.log10(np.maximum(values, 1e-15)) if nmse_in_db else values
        sorted_vals = np.sort(to_plot)
        cdf = np.arange(1, sorted_vals.size + 1, dtype=float) / sorted_vals.size
        legend_label = method_label if method_label not in plotted_methods else None

        ax.plot(
            sorted_vals,
            cdf,
            linewidth=2.0,
            color=color,
            linestyle="-",
            label=legend_label,
        )
        plotted_methods.add(method_label)

    ellipse_specs = {
        "high_mobility": {
            "xy": (0.020, 0.70),
            "width": 0.030,
            "height": 0.075,
            "text_xy": (0.024, 0.65),
        },
        "higher_mobility": {
            "xy": (0.11, 0.70),
            "width": 0.075,
            "height": 0.15,
            "text_xy": (0.055, 0.825),
        },
        "highest_mobility": {
            "xy": (0.235, 0.705),
            "width": 0.125,
            "height": 0.16,
            "text_xy": (0.285, 0.65),
        },
    }

    for mobility, spec in ellipse_specs.items():
        ellipse = Ellipse(
            xy=spec["xy"],
            width=spec["width"],
            height=spec["height"],
            angle=0.0,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )
        ax.add_patch(ellipse)

        ax.text(
            spec["text_xy"][0],
            spec["text_xy"][1],
            mobility_display.get(mobility, mobility),
            ha="left",
            va="top",
            fontsize=10,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, length=5)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5)

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l and l not in unique:
            unique[l] = h

    ax.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        loc="lower right",
        handlelength=2.0,
        fontsize=10,
    )

    fig.tight_layout(pad=0.2)
    _save_cdf_figure(fig, output_path, "chan_pred_nmse_cdf_ellipse")

    # ------------------------------------------------------------------
    # Version 2: Line-style mobility figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    plotted_labels = set()

    for combined_label, values in nmse_values.items():
        if values.size == 0:
            continue

        method_label, mobility = _parse_combined_label(combined_label)
        linestyle, mobility_label = mobility_styles.get(mobility, ("-", mobility))
        color = method_colors.get(method_label, None)

        to_plot = 10.0 * np.log10(np.maximum(values, 1e-15)) if nmse_in_db else values
        sorted_vals = np.sort(to_plot)
        cdf = np.arange(1, sorted_vals.size + 1, dtype=float) / sorted_vals.size

        legend_label = f"{method_label}, {mobility_label}"

        ax.plot(
            sorted_vals,
            cdf,
            linewidth=2.0,
            color=color,
            linestyle=linestyle,
            label=legend_label if legend_label not in plotted_labels else None,
        )
        plotted_labels.add(legend_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, length=5)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5)

    ax.legend(
        frameon=False,
        loc="lower right",
        handlelength=2.0,
        fontsize=10,
    )

    fig.tight_layout(pad=0.2)
    _save_cdf_figure(fig, output_path, "chan_pred_nmse_cdf_linestyle")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=SCRIPT_DIR / "channels_multiple_mu_mimo",
        help=(
            "Root directory containing per-drop results."
        ),
    )
    # parser.add_argument("--mobility", default="high_mobility")
    parser.add_argument(
        "--mobilities",
        nargs="+",
        # default=["high_mobility", "higher_mobility", "highest_mobility"],
        default=["higher_mobility"],
        help=(
            "One or more mobility folder names (e.g., high_mobility higher_mobility). "
            "If provided, this takes precedence over --mobility."
        ),
    )
    parser.add_argument(
        "--drops",
        type=int,
        nargs="+",
        # default=[1, 2, 3, 5],
        default=list(range(1, 21)),
        help="Drop indices to average over (e.g., 1 2 3).",
    )
    parser.add_argument(
        "--rx-ues",
        type=int,
        nargs="+",
        default=[4],
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
        "--modulation-order",
        type=int,
        nargs="+",
        default=[4],
        help="Modulation orders that were simulated (e.g., 2 for QPSK, 4 for 16-QAM).",
    )
    parser.add_argument(
        "--code-rate",
        type=_float_or_fraction,
        nargs="+",
        default=[_float_or_fraction("1/2")],
        help="Code rates that were simulated (accepts fractions like 1/2).",
    )
    parser.add_argument("--quantization", action="store_true", default=True)
    parser.add_argument("--no-quantization", dest="quantization", action="store_false")
    parser.add_argument("--link-adapt", action="store_true", default=True)
    parser.add_argument("--no-link-adapt", dest="link_adapt", action="store_false")
    parser.add_argument(
        "--sync-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--sync-frequency-std-ppb", type=float, default=0.0)
    parser.add_argument("--sync-initial-timing-std-ps", type=float, default=0.0)
    parser.add_argument("--sync-initial-phase-std-deg", type=float, default=0.0)
    parser.add_argument("--sync-phase-noise-s100-dbchz", type=float, default=None)
    parser.add_argument(
        "--feedback-delay-ms",
        type=float,
        choices=[4.0, 8.0],
        default=4.0,
    )
    parser.add_argument("--nmse-in-db", action="store_true", default=False)
    parser.add_argument(
        "--wesn-lite-readout-mode",
        default="centered_ridge",
        help="Select only WESN-Lite artifacts generated with this readout mode.",
    )
    parser.add_argument(
        "--outlier-lower-percentile",
        type=float,
        default=0.0,
        help="Lower percentile bound used to remove outliers (0 disables lower clipping).",
    )
    parser.add_argument(
        "--outlier-upper-percentile",
        type=float,
        default=98.0,
        help="Upper percentile bound used to remove outliers (100 disables upper clipping).",
    )
    parser.add_argument(
        "--output-dir",
        default=SCRIPT_DIR / "plots",
        help="Directory to save the generated plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.outlier_lower_percentile < args.outlier_upper_percentile <= 100.0):
        raise ValueError(
            "--outlier-lower-percentile and --outlier-upper-percentile must satisfy "
            "0 <= lower < upper <= 100"
        )

    base_dir = _resolve_path(args.base_dir, SCRIPT_DIR)
    output_path = _resolve_path(args.output_dir, SCRIPT_DIR) / (
        f"fb_delay_ms_{token(args.feedback_delay_ms)}"
    )
    mobilities = args.mobilities if args.mobilities else [args.mobility]

    scenarios = [
        # Scenario(label="Two-Mode WESN", prediction=True, prediction_method="two_mode"),
        Scenario(label="Kalman Filter", prediction=True, prediction_method="kalman_filter"),
        Scenario(
            label="Steady-State KF",
            prediction=True,
            prediction_method="steady_state_kalman_filter",
        ),
        # Scenario(label="Configured WESN", prediction=True, prediction_method="configured_wesn"),
        Scenario(
            label="Balanced Configured WESN",
            prediction=True,
            prediction_method="configured_wesn_balanced",
        ),
        Scenario(
            label="Balanced Configured WESN-Lite",
            prediction=True,
            prediction_method="configured_wesn_balanced_lite",
        ),
        # Scenario(
        #     label="Low-Rank Configured WESN",
        #     prediction=True,
        #     prediction_method="wesn_lite",
        # ),
        # Scenario(label="ChannelMamba", prediction=True, prediction_method="channelmamba"),
    ]

    prefixes = [
        _build_filename_prefix(
            link_adapt=args.link_adapt,
            mod_order=mod_order,
            code_rate=code_rate,
            rx_ues=rx_ues,
            tx_ues=tx_ues,
        )
        for rx_ues, tx_ues, mod_order, code_rate in product(
            args.rx_ues, args.tx_ues, args.modulation_order, args.code_rate
        )
    ]

    nmse_values: Dict[str, np.ndarray] = {}
    for mobility in mobilities:
        per_mobility_nmse = _collect_nmse(
            base_dir=base_dir,
            mobility=mobility,
            drops=args.drops,
            prefixes=prefixes,
            quantization=args.quantization,
            scenarios=scenarios,
            wesn_lite_readout_mode=args.wesn_lite_readout_mode,
            sync_errors=args.sync_errors,
            sync_frequency_std_ppb=args.sync_frequency_std_ppb,
            sync_initial_timing_std_ps=args.sync_initial_timing_std_ps,
            sync_initial_phase_std_deg=args.sync_initial_phase_std_deg,
            sync_phase_noise_s100_dbchz=args.sync_phase_noise_s100_dbchz,
            feedback_delay_ms=args.feedback_delay_ms,
        )
        per_mobility_nmse = _apply_outlier_filter(
            nmse_values=per_mobility_nmse,
            lower_percentile=args.outlier_lower_percentile,
            upper_percentile=args.outlier_upper_percentile,
        )
        for scenario_label, values in per_mobility_nmse.items():
            nmse_values[f"{scenario_label} ({mobility})"] = values

    summary = ", ".join(f"{k}: {v.size} samples" for k, v in nmse_values.items())
    print(f"Loaded NMSE sample counts -> {summary}")

    _plot_cdf(
        nmse_values=nmse_values,
        nmse_in_db=args.nmse_in_db,
        title=(
            f"MU-MIMO channel prediction NMSE comparison "
            f"({', '.join(mobilities)}, rx={args.rx_ues}, tx={args.tx_ues})"
        ),
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
