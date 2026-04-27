"""Plot channel-prediction NMSE CDFs for MU-MIMO experiments.

This script loads ``chan_pred_nmse`` arrays saved in MU-MIMO result ``.npz`` files
and compares three scenarios:

* no prediction
* Wiener filter prediction
* two-mode prediction

By default it searches under ``results/channels_multiple_mu_mimo`` and expects
subfolders named ``channels_<mobility>_<drop>``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


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
    else:
        patterns = [
            f"{prefix}_perfect_CSI_False_pmi_quantization_{quant_str}.npz",
            f"{prefix}_perfect_CSI_False.npz",
        ]

    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(drop_folder.glob(pattern)))
    return candidates


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
    prefix: str,
    quantization: bool,
    scenarios: Sequence[Scenario],
) -> Dict[str, np.ndarray]:
    values_by_label: Dict[str, List[np.ndarray]] = {s.label: [] for s in scenarios}

    for drop in drops:
        drop_folder = base_dir / f"channels_{mobility}_{drop}"
        for scenario in scenarios:
            candidates = _candidate_paths(drop_folder, prefix, quantization, scenario)
            if not candidates:
                print(
                    f"[WARN] Missing file for scenario={scenario.label}, drop={drop}, "
                    f"folder={drop_folder}"
                )
                continue

            chosen = candidates[0]
            try:
                nmse = _load_nmse_from_npz(chosen)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed loading {chosen}: {exc}")
                continue

            if nmse.size == 0:
                print(
                    f"[WARN] Empty chan_pred_nmse in scenario={scenario.label}, drop={drop}, "
                    f"file={chosen}"
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

def _plot_cdf(
    nmse_values: Dict[str, np.ndarray],
    nmse_in_db: bool,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    # Same color for a prediction method across mobilities
    method_colors = {
        "Two-mode WESN": "tab:blue",
        "Configured WESN": "tab:green",
        "Kalman Filter": "tab:orange",
    }

    # Paper-friendly mobility labels
    mobility_display = {
        "high_mobility": "10 km/h",
        "higher_mobility": "40 km/h",
        "highest_mobility": "80 km/h",
    }

    # Manual ellipse settings so you can control the exact look/placement
    # Format:
    # mobility: {
    #   "xy": (x_center, y_center),
    #   "width": ...,
    #   "height": ...,
    #   "text_xy": (x_text, y_text),
    # }
    ellipse_specs = {
        "high_mobility": {
            "xy": (0.020, 0.70),
            "width": 0.030,   # wider than before
            "height": 0.075,   # flatter / more horizontal
            "text_xy": (0.029, 0.65),  # bottom-right, outside ellipse
        },
        "higher_mobility": {
            "xy": (0.11, 0.7),
            "width": 0.075,
            "height": 0.15,
            "text_xy": (0.055, 0.8),   # bottom-right, outside ellipse
        },
        "highest_mobility": {
            "xy": (0.235, 0.705),
            "width": 0.125,
            "height": 0.16,
            "text_xy": (0.285, 0.65),   # bottom-right, outside ellipse
        },
    }

    for combined_label, values in nmse_values.items():
        if values.size == 0:
            continue

        # Expect labels like: "Kalman Filter (high_mobility)"
        if " (" in combined_label and combined_label.endswith(")"):
            method_label, mobility = combined_label.rsplit(" (", 1)
            mobility = mobility[:-1]
        else:
            method_label = combined_label
            mobility = "high_mobility"

        color = method_colors.get(method_label, None)

        to_plot = 10.0 * np.log10(np.maximum(values, 1e-15)) if nmse_in_db else values
        sorted_vals = np.sort(to_plot)
        cdf = np.arange(1, sorted_vals.size + 1, dtype=float) / sorted_vals.size

        # Same linestyle for all curves
        ax.plot(
            sorted_vals,
            cdf,
            linewidth=2.2,
            color=color,
            linestyle="-",
            label=method_label if mobility == "high_mobility" else None,
        )

    xlabel = "Channel prediction NMSE (dB)" if nmse_in_db else "Channel prediction NMSE"
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("CDF", fontsize=14)
    ax.grid(alpha=0.25)

    # Add dashed ellipses and external labels
    for mobility, spec in ellipse_specs.items():
        ellipse = Ellipse(
            xy=spec["xy"],
            width=spec["width"],
            height=spec["height"],
            angle=0.0,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            linestyle="--",   # dashed ellipse
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

    # Only keep prediction-method legend
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l and l not in unique:
            unique[l] = h

    ax.legend(
        unique.values(),
        unique.keys(),
        loc="lower right",
        frameon=True,
    )

    plt.tight_layout()

    output_path.mkdir(parents=True, exist_ok=True)
    save_file = output_path / "chan_pred_nmse_cdf.png"
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {save_file}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=SCRIPT_DIR / "channels_multiple_mu_mimo",
        help=(
            "Root directory containing per-drop results."
        ),
    )
    parser.add_argument("--mobility", default="high_mobility")
    parser.add_argument(
        "--mobilities",
        nargs="+",
        default=["high_mobility", "higher_mobility", "highest_mobility"],
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
        default=4,
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
    parser.add_argument("--nmse-in-db", action="store_true", default=False)
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
    output_path = _resolve_path(args.output_dir, SCRIPT_DIR)
    mobilities = args.mobilities if args.mobilities else [args.mobility]

    scenarios = [
        # Scenario(label="Two-mode WESN", prediction=True, prediction_method="two_mode"),
        Scenario(label="Configured WESN", prediction=True, prediction_method="configured_wesn"),
        Scenario(label="Kalman Filter", prediction=True, prediction_method="kalman_filter"),
    ]

    prefix = _build_filename_prefix(
        link_adapt=args.link_adapt,
        mod_order=args.modulation_order,
        code_rate=args.code_rate,
        rx_ues=args.rx_ues,
        tx_ues=args.tx_ues,
    )

    nmse_values: Dict[str, np.ndarray] = {}
    for mobility in mobilities:
        per_mobility_nmse = _collect_nmse(
            base_dir=base_dir,
            mobility=mobility,
            drops=args.drops,
            prefix=prefix,
            quantization=args.quantization,
            scenarios=scenarios,
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