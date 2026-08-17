"""Plot MU-MIMO KPI sweeps aggregated over multiple drops.

The plotting logic is tailored for the artifacts produced by
``sims/run_sim_mu_mimo_kpi_multiple_drops.sh``.  The script loads the
per-drop ``npz`` files, averages metrics across drops, and produces four
figures, plus two residual-synchronization sweeps at fixed RX/TX counts:

* Uncoded BER vs. number of RUs (fixed UEs, fixed MCS)
* Uncoded BER vs. number of UEs (fixed RUs, fixed MCS)
* Throughput vs. number of RUs (fixed UEs, best MCS per point)
* Throughput vs. number of UEs (fixed RUs, best MCS per point)
* Throughput vs. residual fractional-frequency standard deviation
* Throughput vs. initial timing-offset standard deviation
* Throughput vs. oscillator phase-noise level

For BER plots, the modulation order and code rate are fixed by the
command-line arguments.  For throughput plots, the script selects, for
each data point, the MCS that maximizes the *average* throughput across
all requested drops and prints the maximizing MCS choices.

The script also plots multiple channel prediction baselines, including the
full Kalman filter, drop-configured steady-state Kalman filter, configured
WESN, and ChannelMamba. By default, outputs are expected directly under::

    results/channels_multiple_mu_mimo/channels_<mobility>_<drop>

"""

from __future__ import annotations

import argparse
import glob
import json

import os

from pathlib import Path
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SYNC_RESULT_MODEL_VERSION = "clock_v2"
PA_RESULT_MODEL_VERSION = "rapp_v1"

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
    link_adapt: bool
    scenarios: Sequence["Scenario"]
    channelmamba_seen_drops: Sequence[int]
    channelmamba_all_drops: Sequence[int]
    wesn_lite_readout_mode: str = "centered_ridge"
    sync_errors: bool = False
    sync_frequency_std_ppb: float = 0.0
    sync_initial_timing_std_ps: float = 0.0
    sync_initial_phase_std_deg: float = 0.0
    sync_phase_noise_s100_dbchz: Optional[float] = None
    feedback_delay_ms: float = 4.0
    pa_enabled: bool = False
    pa_ibo_db: float = 6.5
    pa_rho: float = 3.0
    pa_model_version: str = PA_RESULT_MODEL_VERSION


def _filename_token(value: float) -> str:
    return format(float(value), "g").replace("-", "m").replace(".", "p")


def sync_result_suffix(
    frequency_std_ppb: float,
    initial_timing_std_ps: float,
    initial_phase_std_deg: float,
    phase_noise_s100_dbchz: Optional[float],
) -> str:
    pn_token = (
        "off"
        if phase_noise_s100_dbchz is None
        else _filename_token(phase_noise_s100_dbchz)
    )
    return (
        f"_sync_{SYNC_RESULT_MODEL_VERSION}"
        f"_freq_std_ppb_{_filename_token(frequency_std_ppb)}"
        f"_timing0_std_ps_{_filename_token(initial_timing_std_ps)}"
        f"_phase0_std_deg_{_filename_token(initial_phase_std_deg)}"
        f"_pn_s100_dbchz_{pn_token}"
    )


def pa_result_suffix(
    enabled: bool,
    ibo_db: float,
    rho: float,
    model_version: str,
) -> str:
    if not enabled:
        return ""
    return (
        f"_pa_{model_version}"
        f"_ibo_db_{_filename_token(ibo_db)}"
        f"_rho_{_filename_token(rho)}"
    )


def feedback_delay_result_suffix(feedback_delay_ms: float) -> str:
    return f"_fb_delay_ms_{_filename_token(feedback_delay_ms)}"


def phase_noise_rms_deg(
    s100_dbchz: float,
    *,
    duration_s: float,
) -> float:
    """RMS Wiener phase innovation over ``duration_s`` (Ngo--Larsson Eq. 23)."""

    variance_rad2 = (
        4.0
        * np.pi**2
        * (100e3) ** 2
        * 10.0 ** (float(s100_dbchz) / 10.0)
        * float(duration_s)
    )
    return float(np.rad2deg(np.sqrt(variance_rad2)))

def _resolve_path(path: str, relative_to: Path) -> Path:
    """Resolve ``path`` against ``relative_to`` when not absolute."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (relative_to / candidate).resolve()

################################################################################
# Data loading helpers
################################################################################


@dataclass
class DataPoint:
    uncoded_ber: float
    coded_ber: float
    throughput: float
    channel_prediction_nmse: float = float("nan")


@dataclass(frozen=True)
class SyncThroughputStatistics:
    """Mean throughput and paired 95% confidence half-widths."""

    scenario: "Scenario"
    means: Sequence[float]
    ci95: Sequence[float]
    counts: Sequence[int]


@dataclass(frozen=True)
class PaMetricStatistics:
    """Per-IBO means and paired 95% confidence half-widths."""

    scenario: "Scenario"
    means: Sequence[float]
    ci95: Sequence[float]
    counts: Sequence[int]

@dataclass(frozen=True)
class Scenario:
    perfect_csi: bool
    prediction: bool
    quantization: bool
    label: str
    link_adapt: bool = False
    prediction_method: Optional[str] = None


class ResultLoader:
    def __init__(self, cfg: PlotConfig) -> None:
        self.cfg = cfg

    def _drop_folder(self, drop_id: int, scenario: Scenario) -> str:
        folder_name = f"channels_{self.cfg.mobility}_{drop_id}"
        return os.path.join(self.cfg.base_dir, folder_name)

    @staticmethod
    def _result_prefix(
        rx_ues: int,
        tx_ues: int,
        mod_order: int,
        code_rate: object,
        scenario: Scenario,
    ) -> str:
        if scenario.link_adapt:
            experiment = f"link_adapt_rx_UE_{rx_ues}_tx_UE_{tx_ues}"
        else:
            experiment = (
                f"mod_order_{mod_order}_code_rate_{code_rate}_"
                f"rx_UE_{rx_ues}_tx_UE_{tx_ues}"
            )
        return f"mu_mimo_results_{experiment}"
    
    @staticmethod
    def _parse_code_rate_from_path(path: str) -> Optional[float]:
        basename = os.path.basename(path)
        try:
            middle = basename.split("code_rate_")[1]
            code_rate_str = middle.split("_rx_UE")[0]
            return float(code_rate_str)
        except (IndexError, ValueError):
            return None
        
    @staticmethod
    def _append_suffix(path: str, suffix: str) -> str:
        if not suffix:
            return path
        if path.endswith(".npz"):
            return f"{path[:-4]}{suffix}.npz"
        return f"{path}{suffix}"

    def _suffixes_for_scenario(self, scenario: Scenario) -> List[str]:
        if scenario.prediction_method in (
            "configured_wesn",
            "configured_wesn_balanced",
            "configured_wesn_balanced_lite",
            "wesn_lite",
        ):
            # Backward compatibility: early time-split runs had no suffix.
            return [
                "_time_split_readout_*_workers_*",
                "_time_split_readout_*",
                "_time_split_workers_*",
                "_time_split",
                "",
            ]
        if scenario.prediction_method == "channelmamba":
            return ["_time_split_workers_*", "_time_split"]
        if scenario.prediction:
            # Predictor parallelism is encoded in current result filenames for
            # every prediction method, including the full and steady-state KFs.
            return ["_workers_*", ""]
        return [""]
    
    def _prediction_patterns(
        self,
        prefix: str,
        scenario: Scenario,
    ) -> List[str]:
        quant_str = str(scenario.quantization)
        method = scenario.prediction_method
        patterns = []
        if method:
            patterns.append(
                f"{prefix}_prediction_{method}_pmi_quantization_{quant_str}.npz"
            )
            patterns.append(
                f"{prefix}_prediction_{method}_pmi_quantization_{quant_str}_imitation_none_steps_0.npz"
            )
        else:
            patterns.append(
                f"{prefix}_prediction_two_mode_pmi_quantization_{quant_str}.npz"
            )
            patterns.append(f"{prefix}_prediction.npz")
        return patterns

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
        suffixes = self._suffixes_for_scenario(scenario)
        sync_suffix = sync_result_suffix(
            self.cfg.sync_frequency_std_ppb,
            self.cfg.sync_initial_timing_std_ps,
            self.cfg.sync_initial_phase_std_deg,
            self.cfg.sync_phase_noise_s100_dbchz,
        )
        pa_suffix = pa_result_suffix(
            self.cfg.pa_enabled,
            self.cfg.pa_ibo_db,
            self.cfg.pa_rho,
            self.cfg.pa_model_version,
        )
        feedback_suffix = feedback_delay_result_suffix(
            self.cfg.feedback_delay_ms
        )
        result_suffixes = [sync_suffix + pa_suffix + feedback_suffix]
        # Legacy artifacts without a feedback-delay suffix all used the
        # original 4 ms cycle. Never use them for an 8 ms request.
        if np.isclose(self.cfg.feedback_delay_ms, 4.0):
            result_suffixes.append(sync_suffix + pa_suffix)
        for pattern in patterns:
            for suffix in suffixes:
                for result_suffix in result_suffixes:
                    suffixed_pattern = self._append_suffix(
                        pattern, suffix + result_suffix
                    )
                    full_pattern = os.path.join(folder, suffixed_pattern)
                    if "*" in suffixed_pattern:
                        matches = glob.glob(full_pattern)
                        matches.sort()
                        candidates.extend(matches)
                    else:
                        candidates.append(full_pattern)
        # Keep old zero-error artifacts readable while preferring the new,
        # explicitly parameterized filenames above.
        if (
            not self.cfg.sync_errors
            and not self.cfg.pa_enabled
            and self.cfg.sync_frequency_std_ppb == 0
            and self.cfg.sync_initial_timing_std_ps == 0
            and self.cfg.sync_initial_phase_std_deg == 0
            and self.cfg.sync_phase_noise_s100_dbchz is None
            and not any(os.path.exists(candidate) for candidate in candidates)
        ):
            for pattern in patterns:
                for suffix in suffixes:
                    legacy_pattern = self._append_suffix(pattern, suffix)
                    full_pattern = os.path.join(folder, legacy_pattern)
                    if "*" in legacy_pattern:
                        candidates.extend(
                            path
                            for path in sorted(glob.glob(full_pattern))
                            if "_sync_clock_" not in os.path.basename(path)
                        )
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
        folder = self._drop_folder(drop_id, scenario)
        prefix = self._result_prefix(
            rx_ues, tx_ues, mod_order, str(code_rate), scenario
        )
        existing_candidates = [
            candidate
            for candidate in self._candidate_paths(folder, prefix, scenario)
            if os.path.exists(candidate)
        ]
        if scenario.prediction_method == "wesn_lite":
            matching_candidates = [
                candidate
                for candidate in existing_candidates
                if self._wesn_lite_readout_mode(candidate)
                == self.cfg.wesn_lite_readout_mode
            ]
            if matching_candidates:
                # Prefer the latest artifact when several worker-count variants
                # implement the same requested algorithm.
                return max(matching_candidates, key=os.path.getmtime)
            return None
        if existing_candidates:
            # Non-WESN predictors can have both a legacy unsuffixed artifact
            # and a current worker-suffixed artifact. Select the latest run.
            if scenario.prediction_method not in ("configured_wesn",):
                return max(existing_candidates, key=os.path.getmtime)
            return existing_candidates[0]
            
        if scenario.link_adapt:
            return None

        # As a fallback, try to match slightly different code-rate strings.
        pattern = os.path.join(
            folder,
            self._result_prefix(rx_ues, tx_ues, mod_order, "*", scenario)
            + "*.npz",
        )
        matches = glob.glob(pattern)
        required_sync_suffix = sync_result_suffix(
            self.cfg.sync_frequency_std_ppb,
            self.cfg.sync_initial_timing_std_ps,
            self.cfg.sync_initial_phase_std_deg,
            self.cfg.sync_phase_noise_s100_dbchz,
        )
        required_pa_suffix = pa_result_suffix(
            self.cfg.pa_enabled,
            self.cfg.pa_ibo_db,
            self.cfg.pa_rho,
            self.cfg.pa_model_version,
        )
        required_feedback_suffix = feedback_delay_result_suffix(
            self.cfg.feedback_delay_ms
        )
        matches = [
            path
            for path in matches
            if required_sync_suffix in path
            and required_pa_suffix in path
            and (
                required_feedback_suffix in path
                or (
                    np.isclose(self.cfg.feedback_delay_ms, 4.0)
                    and "_fb_delay_ms_" not in path
                )
            )
        ]
        if not matches:
            return None

        target = float(code_rate)
        matches.sort(
            key=lambda path: abs(
                (self._parse_code_rate_from_path(path) or target) - target
            )
        )
        for match in matches:
            basename = os.path.basename(match)
            if scenario.prediction:
                method = scenario.prediction_method
                if method and f"_prediction_{method}_" in basename:
                    return match
                if method is None and "prediction" in basename:
                    return match
            if not scenario.prediction and "perfect_CSI" in os.path.basename(match):
                return match
        return None

    @staticmethod
    def _wesn_lite_readout_mode(path: str) -> Optional[str]:
        """Read the algorithm mode from explicit or legacy result metadata."""
        try:
            with np.load(path, allow_pickle=True) as data:
                explicit = data.get("wesn_lite_readout_mode")
                if explicit is not None:
                    return str(np.asarray(explicit).reshape(-1)[0]).replace(
                        "-", "_"
                    )
                raw = data.get("predictor_complexity_raw_json")
                if raw is None:
                    return None
                metrics = json.loads(str(np.asarray(raw).item()))
                modes = {
                    str(value.get("readout_objective", "")).replace("-", "_")
                    for value in metrics.get(
                        "per_link_predictor_phases", {}
                    ).values()
                    if value.get("readout_objective")
                }
                return next(iter(modes)) if len(modes) == 1 else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

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
    
    @staticmethod
    def _coded_ber_from_npz(data: np.lib.npyio.NpzFile) -> float:
        coded = data.get("ldpc_ber_list")
        if coded is None:
            return float("nan")
        coded_array = np.asarray(coded, dtype=float)
        return float(np.nanmean(coded_array))

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
            coded_ber = self._coded_ber_from_npz(data)
            throughput = self._scalar_from_array(np.atleast_1d(data.get("throughput", [])))
            nmse_raw = data.get("chan_pred_nmse")
            channel_prediction_nmse = (
                float(np.nanmean(np.asarray(nmse_raw, dtype=float)))
                if nmse_raw is not None and np.asarray(nmse_raw).size
                else float("nan")
            )
        return DataPoint(
            uncoded_ber=uncoded_ber,
            coded_ber=coded_ber,
            throughput=throughput,
            channel_prediction_nmse=channel_prediction_nmse,
        )



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
    drops: Optional[Iterable[int]] = None,
) -> Dict[Tuple[Scenario, int, int, Optional[int], Optional[float]], List[DataPoint]]:
    results: Dict[
        Tuple[Scenario, int, int, Optional[int], Optional[float]], List[DataPoint]
    ] = {}
    drop_iterable = list(loader.cfg.drops if drops is None else drops)
    for scenario in scenarios:
        for drop_id in drop_iterable:
            for rx_ues in rx_values:
                for tx_ues in tx_values:
                    if scenario.link_adapt:
                        datapoint = loader.load_datapoint(
                            drop_id, rx_ues, tx_ues, 0, 0, scenario
                        )
                        if datapoint is None:
                            continue
                        results.setdefault(
                            (scenario, rx_ues, tx_ues, None, None), []
                        ).append(datapoint)
                        continue
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
    aggregated: Dict[Tuple[Scenario, int, int, Optional[int], Optional[float]], List[DataPoint]],
    scenario: Scenario,
    rx_ues: int,
    tx_ues: int,
    mod_order: Optional[int],
    code_rate: Optional[float],
) -> Optional[DataPoint]:
    if scenario.link_adapt:
        key = (scenario, rx_ues, tx_ues, None, None)
    else:
        key = (scenario, rx_ues, tx_ues, mod_order, float(code_rate))
    points = aggregated.get(key, [])
    return average_datapoints(points) if points else None

def average_datapoints(points: Sequence[DataPoint]) -> DataPoint:
    nmse_values = np.asarray(
        [p.channel_prediction_nmse for p in points], dtype=np.float64
    )
    mean_nmse = (
        float(np.nanmean(nmse_values))
        if np.any(np.isfinite(nmse_values))
        else float("nan")
    )
    return DataPoint(
        uncoded_ber=float(np.nanmean([p.uncoded_ber for p in points])),
        coded_ber=float(np.nanmean([p.coded_ber for p in points])),
        throughput=float(np.nanmean([p.throughput for p in points])),
        channel_prediction_nmse=mean_nmse,
    )

def _default_scenarios(
    include_prediction: bool = True, link_adapt: bool = False
) -> List[Scenario]:
    scenarios = [
        # Scenario(
        #     perfect_csi=False,
        #     prediction=True,
        #     quantization=True,
        #     label="ChannelMamba",
        #     link_adapt=link_adapt,
        #     prediction_method="channelmamba",
        # ),
        # Scenario(
        #     perfect_csi=False,
        #     prediction=False,
        #     quantization=True,
        #     label="Outdated CSI",
        #     link_adapt=link_adapt,
        # ),
        # Scenario(
        #     perfect_csi=False,
        #     prediction=True,
        #     quantization=True,
        #     label="Two-Mode WESN",
        #     prediction_method="two_mode",
        #     link_adapt=link_adapt,
        # ),
        Scenario(
            perfect_csi=False,
            prediction=True,
            quantization=True,
            label="Kalman Filter",
            link_adapt=link_adapt,
            prediction_method="kalman_filter",
        ),
        Scenario(
            perfect_csi=False,
            prediction=True,
            quantization=True,
            label="Steady-State KF",
            link_adapt=link_adapt,
            prediction_method="steady_state_kalman_filter",
        ),
        # Scenario(
        #     perfect_csi=False,
        #     prediction=True,
        #     quantization=True,
        #     label="Configured WESN",
        #     link_adapt=link_adapt,
        #     prediction_method="configured_wesn",
        # ),
        Scenario(
            perfect_csi=False,
            prediction=True,
            quantization=True,
            label="Balanced Configured WESN",
            link_adapt=link_adapt,
            prediction_method="configured_wesn_balanced",
        ),
        Scenario(
            perfect_csi=False,
            prediction=True,
            quantization=True,
            label="Balanced Configured WESN-Lite",
            link_adapt=link_adapt,
            prediction_method="configured_wesn_balanced_lite",
        ),
        # Scenario(
        #     perfect_csi=False,
        #     prediction=True,
        #     quantization=True,
        #     label="Low-Rank Configured WESN",
        #     link_adapt=link_adapt,
        #     prediction_method="wesn_lite",
        # ),
        # Scenario(
        #     perfect_csi=True,
        #     prediction=False,
        #     quantization=True,
        #     label="Perfect Prediction",
        #     link_adapt=link_adapt,
        # ),
        # Scenario(
        #     perfect_csi=True,
        #     prediction=False,
        #     quantization=False,
        #     label="Perfect CSI at BS (no quantization, no delay)",
        #     link_adapt=link_adapt,
        # ),
    ]

    if include_prediction:
        return scenarios
    return [scenario for scenario in scenarios if not scenario.prediction]

# Different marker styles for different curves
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
    "Two-Mode WESN": {
        "color": "tab:blue",
        "marker": "o",
        "label": "Two-Mode WESN",
    },
    "Kalman Filter": {
        "color": "tab:orange",
        "marker": "s",
        "label": "Kalman Filter",
    },
    "Steady-State KF": {
        "color": "tab:purple",
        "marker": "P",
        "linestyle": "--",
        "label": "Steady-State KF",
    },
    "Configured WESN": {
        "color": "tab:green",
        "marker": "^",
        "label": "Configured WESN",
    },
    "Balanced Configured WESN": {
        "color": "tab:cyan",
        "marker": "X",
        "label": "Balanced Configured WESN",
    },
    "Balanced Configured WESN-Lite": {
        "color": "tab:pink",
        "marker": "*",
        "label": "Balanced Configured WESN-Lite",
    },
    "Low-Rank Configured WESN": {
        "color": "tab:brown",
        "marker": "v",
        "linestyle": "-.",
        "label": "Low-Rank Configured WESN",
    },
    "ChannelMamba": {
        "color": "tab:red",
        "marker": "D",
        "label": "ChannelMamba",
    },
    "Outdated CSI": {
        "color": "0.45",
        "marker": "x",
        "label": "Outdated CSI",
    },
}

def _style_for_label(label: str) -> dict:
    return STYLE.get(label, {"marker": "o", "label": label})

################################################################################
# Plotting helpers
################################################################################

def _save_figure_multi_format(output_path: str) -> None:
    """
    Save paper-ready vector figures plus a PNG preview.
    Prefer PDF for LaTeX/IEEE papers.
    """
    base, _ = os.path.splitext(output_path)

    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base + ".svg", bbox_inches="tight")

def plot_metric(
    x_values: Sequence[float],
    series: Sequence[Tuple[str, Sequence[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
    series_errors: Optional[Sequence[Tuple[str, Sequence[float]]]] = None,
    x_tick_labels: Optional[Sequence[str]] = None,
    top_tick_labels: Optional[Sequence[str]] = None,
    top_xlabel: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    errors_by_label = dict(series_errors or [])

    for label, y_values in series:
        st = _style_for_label(label)
        plot_kwargs = dict(
            marker=st["marker"],
            color=st.get("color", None),
            linestyle=st.get("linestyle", "-"),
            linewidth=2.0,
            markersize=6.0,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=st["label"],
        )
        if label in errors_by_label:
            ax.errorbar(
                x_values,
                y_values,
                yerr=errors_by_label[label],
                capsize=2.0,
                **plot_kwargs,
            )
        else:
            ax.plot(x_values, y_values, **plot_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_tick_labels is not None:
        ax.set_xticks(x_values, labels=x_tick_labels)
    if top_tick_labels is not None:
        top_ax = ax.twiny()
        top_ax.set_xlim(ax.get_xlim())
        top_ax.set_xticks(x_values, labels=top_tick_labels)
        top_ax.tick_params(direction="in", length=5)
        if top_xlabel is not None:
            top_ax.set_xlabel(top_xlabel)
    # ax.set_ylim(top=27)
    ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, length=5)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5)

    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        fontsize=9,
        handlelength=1.8,
        borderaxespad=0.2,
    )
    fig.tight_layout(pad=0.2)

    _save_figure_multi_format(output_path)
    print(f"Saved: {output_path}")

def semilogy_metric(
    x_values: Sequence[float],
    series: Sequence[Tuple[str, Sequence[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    for label, y_values in series:
        st = _style_for_label(label)
        ax.semilogy(
            x_values,
            y_values,
            marker=st["marker"],
            color=st.get("color", None),
            linestyle=st.get("linestyle", "-"),
            linewidth=2.0,
            markersize=6.0,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=st["label"],
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="-", linewidth=0.35, alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, length=5)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5)

    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=10,
        handlelength=1.8,
        borderaxespad=0.2,
    )
    fig.tight_layout(pad=0.2)

    _save_figure_multi_format(output_path)
    print(f"Saved: {output_path}")



def select_best_mcs(
    aggregated: Dict[Tuple[Scenario, int, int, Optional[int], Optional[float]], List[DataPoint]],
    scenario: Scenario,
    rx_ues: int,
    tx_ues: int,
    modulation_orders: Iterable[int],
    code_rates: Iterable[float],
) -> Tuple[Optional[float], Optional[Tuple[int, float]]]:
    if scenario.link_adapt:
        datapoint = _average_metric(
            aggregated, scenario, rx_ues, tx_ues, mod_order=None, code_rate=None
        )
        return (datapoint.throughput if datapoint else None), None
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


def sync_throughput_statistics(
    loader_type: type[ResultLoader],
    cfg: PlotConfig,
    scenarios: Sequence[Scenario],
    sweep_values: Sequence[Optional[float]],
    sweep: str,
) -> List[SyncThroughputStatistics]:
    """Build throughput means and paired confidence intervals for a sweep.

    Both UE dimensions are fixed and all non-swept synchronization dimensions
    are zero. For non-baseline points, the confidence interval is computed from
    per-drop differences relative to the all-zero point. ``None`` denotes the
    phase-noise-off baseline and is valid only for the phase-noise sweep.
    """

    if sweep not in {"frequency", "timing", "phase_noise"}:
        raise ValueError("unsupported synchronization sweep")

    samples_by_scenario = {scenario: [] for scenario in scenarios}
    for sweep_value in sweep_values:
        if sweep_value is None and sweep != "phase_noise":
            raise ValueError("only the phase-noise sweep supports an off point")
        numeric_value = 0.0 if sweep_value is None else float(sweep_value)
        phase_noise_value = (
            numeric_value
            if sweep == "phase_noise" and sweep_value is not None
            else None
        )
        point_cfg = replace(
            cfg,
            # Sweep artifacts, including the explicit all-zero baseline, must
            # come from the current versioned experiment. This deliberately
            # disables the legacy zero-result fallback.
            sync_errors=True,
            sync_frequency_std_ppb=(
                numeric_value if sweep == "frequency" else 0.0
            ),
            sync_initial_timing_std_ps=(
                numeric_value if sweep == "timing" else 0.0
            ),
            sync_initial_phase_std_deg=0.0,
            sync_phase_noise_s100_dbchz=phase_noise_value,
        )
        point_loader = loader_type(point_cfg)
        aggregated = aggregate_metrics(
            point_loader,
            scenarios,
            [cfg.fixed_rx_for_tx_sweep],
            [cfg.fixed_tx_for_rx_sweep],
            cfg.modulation_orders,
            cfg.code_rates,
        )
        for scenario in scenarios:
            _, best_mcs = select_best_mcs(
                aggregated,
                scenario,
                cfg.fixed_rx_for_tx_sweep,
                cfg.fixed_tx_for_rx_sweep,
                cfg.modulation_orders,
                cfg.code_rates,
            )
            mod_order, code_rate = best_mcs or (0, 0.0)
            samples = {}
            for drop_id in cfg.drops:
                datapoint = point_loader.load_datapoint(
                    drop_id,
                    cfg.fixed_rx_for_tx_sweep,
                    cfg.fixed_tx_for_rx_sweep,
                    mod_order,
                    code_rate,
                    scenario,
                )
                if datapoint is not None:
                    samples[int(drop_id)] = datapoint.throughput
            samples_by_scenario[scenario].append(samples)

    baseline_value = None if sweep == "phase_noise" else 0.0
    baseline_idx = next(
        (
            idx
            for idx, value in enumerate(sweep_values)
            if value == baseline_value
        ),
        None,
    )
    statistics = []
    for scenario in scenarios:
        point_samples = samples_by_scenario[scenario]
        baseline_samples = (
            point_samples[baseline_idx] if baseline_idx is not None else {}
        )
        means, ci95, counts = [], [], []
        for idx, samples in enumerate(point_samples):
            values = np.asarray(list(samples.values()), dtype=float)
            means.append(float(np.mean(values)) if values.size else np.nan)
            counts.append(int(values.size))
            if baseline_idx is not None and idx != baseline_idx:
                common_drops = sorted(set(samples) & set(baseline_samples))
                uncertainty_samples = np.asarray(
                    [
                        samples[drop_id] - baseline_samples[drop_id]
                        for drop_id in common_drops
                    ],
                    dtype=float,
                )
            else:
                uncertainty_samples = values
            if uncertainty_samples.size > 1:
                half_width = 1.96 * float(
                    np.std(uncertainty_samples, ddof=1)
                    / np.sqrt(uncertainty_samples.size)
                )
            elif uncertainty_samples.size == 1:
                half_width = 0.0
            else:
                half_width = np.nan
            ci95.append(half_width)
        statistics.append(
            SyncThroughputStatistics(scenario, means, ci95, counts)
        )
    return statistics


def sync_throughput_series(
    loader_type: type[ResultLoader],
    cfg: PlotConfig,
    scenarios: Sequence[Scenario],
    sweep_values: Sequence[float],
    sweep: str,
) -> List[Tuple[Scenario, Sequence[float]]]:
    """Backward-compatible mean-only view of synchronization statistics."""

    return [
        (item.scenario, item.means)
        for item in sync_throughput_statistics(
            loader_type, cfg, scenarios, sweep_values, sweep
        )
    ]


def pa_metric_statistics(
    loader_type: type[ResultLoader],
    cfg: PlotConfig,
    scenarios: Sequence[Scenario],
    ibo_values: Sequence[float],
    metric: str,
) -> List[PaMetricStatistics]:
    """Build per-IBO metric means and paired confidence intervals."""

    if metric not in {
        "throughput",
        "uncoded_ber",
        "coded_ber",
        "channel_prediction_nmse",
    }:
        raise ValueError("unsupported PA-sweep metric")
    if not ibo_values:
        return []

    samples_by_scenario = {scenario: [] for scenario in scenarios}
    for ibo_db in ibo_values:
        point_cfg = replace(
            cfg,
            pa_enabled=True,
            pa_ibo_db=float(ibo_db),
        )
        point_loader = loader_type(point_cfg)
        aggregated = aggregate_metrics(
            point_loader,
            scenarios,
            [cfg.fixed_rx_for_tx_sweep],
            [cfg.fixed_tx_for_rx_sweep],
            cfg.modulation_orders,
            cfg.code_rates,
        )
        for scenario in scenarios:
            _, best_mcs = select_best_mcs(
                aggregated,
                scenario,
                cfg.fixed_rx_for_tx_sweep,
                cfg.fixed_tx_for_rx_sweep,
                cfg.modulation_orders,
                cfg.code_rates,
            )
            mod_order, code_rate = best_mcs or (0, 0.0)
            samples = {}
            for drop_id in cfg.drops:
                datapoint = point_loader.load_datapoint(
                    drop_id,
                    cfg.fixed_rx_for_tx_sweep,
                    cfg.fixed_tx_for_rx_sweep,
                    mod_order,
                    code_rate,
                    scenario,
                )
                if datapoint is not None:
                    samples[int(drop_id)] = float(getattr(datapoint, metric))
            samples_by_scenario[scenario].append(samples)

    reference_idx = int(np.argmax(np.asarray(ibo_values, dtype=float)))
    statistics = []
    for scenario in scenarios:
        point_samples = samples_by_scenario[scenario]
        reference_samples = point_samples[reference_idx]
        means, ci95, counts = [], [], []
        for idx, samples in enumerate(point_samples):
            values = np.asarray(list(samples.values()), dtype=float)
            finite_values = values[np.isfinite(values)]
            means.append(
                float(np.mean(finite_values)) if finite_values.size else np.nan
            )
            counts.append(int(finite_values.size))
            if idx == reference_idx:
                uncertainty_samples = finite_values
            else:
                common_drops = sorted(set(samples) & set(reference_samples))
                uncertainty_samples = np.asarray(
                    [
                        samples[drop_id] - reference_samples[drop_id]
                        for drop_id in common_drops
                        if np.isfinite(samples[drop_id])
                        and np.isfinite(reference_samples[drop_id])
                    ],
                    dtype=float,
                )
            if uncertainty_samples.size > 1:
                half_width = 1.96 * float(
                    np.std(uncertainty_samples, ddof=1)
                    / np.sqrt(uncertainty_samples.size)
                )
            elif uncertainty_samples.size == 1:
                half_width = 0.0
            else:
                half_width = np.nan
            ci95.append(half_width)
        statistics.append(PaMetricStatistics(scenario, means, ci95, counts))
    return statistics


################################################################################
# Main plotting routine
################################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=SCRIPT_DIR / "channels_multiple_mu_mimo",
        help=(
            "Root directory containing per-drop results."
        ),
    )
    parser.add_argument("--mobility", default="higher_mobility", help="Mobility string used in the folder names.")
    parser.add_argument(
        "--drops",
        type=int,
        nargs="+",
        # default=[1, 2],
        # default=[3, 13, 14, 15, 19, 20], # good: 20, 3,  okay: 19, 15, not great: 14, 13
        default=list(range(1, 21)),
        help="Drop indices to average over (e.g., 1 2 3).",
    )
    parser.add_argument(
        "--channelmamba-seen-drops",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="Seen-environment set for ChannelMamba curve (default: 1..10).",
    )
    parser.add_argument(
        "--channelmamba-all-drops",
        type=int,
        nargs="+",
        default=list(range(1, 21)),
        help="All-environment set for ChannelMamba curve (default: 1..20).",
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
        "--modulation-orders",
        type=int,
        nargs="+",
        default=[4],
        help="Modulation orders that were simulated (e.g., 2 for QPSK, 4 for 16-QAM).",
    )
    parser.add_argument(
        "--code-rates",
        type=_float_or_fraction,
        nargs="+",
        default=[_float_or_fraction("1/2")],
        help="Code rates that were simulated (accepts fractions like 1/2).",
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
        default=_float_or_fraction("1/2"),
        help="Code rate to use for BER plots (accepts fractions like 1/2).",
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
        "--wesn-lite-readout-mode",
        choices=["matched_ridge", "centered_ridge"],
        default="centered_ridge",
        help="Select only WESN-Lite artifacts generated with this readout mode.",
    )
    parser.add_argument(
        "--sync-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load artifacts generated with residual synchronization errors.",
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
        help="Load results generated with this one-cycle CSI feedback delay.",
    )
    parser.add_argument(
        "--sync-frequency-std-ppb-values",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 3.73, 10.0, 30.0],
        help="Residual fractional-frequency standard deviations in ppb.",
    )
    parser.add_argument(
        "--sync-initial-timing-std-ps-values",
        type=float,
        nargs="+",
        default=[0.0, 30.0, 60.0, 70.0, 200.0],
        help="Initial post-synchronization timing standard deviations in ps.",
    )
    parser.add_argument(
        "--sync-phase-noise-s100-dbchz-values",
        type=float,
        nargs="+",
        default=[-120.0, -110.0, -100.0, -90.0],
        help="Phase-noise spectrum levels in dBc/Hz at 100 kHz offset.",
    )
    parser.add_argument(
        "--sync-phase-innovation-delay-slots",
        type=int,
        default=None,
        help=(
            "Optional CSI-delay override for the phase-innovation secondary "
            "axis. By default it is derived from --feedback-delay-ms."
        ),
    )
    parser.add_argument(
        "--sync-slot-duration-ms",
        type=float,
        default=1.0,
        help="Slot duration used for the phase-innovation secondary axis.",
    )
    parser.add_argument(
        "--plot-pa-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate channel-prediction comparison figures versus PA IBO.",
    )
    parser.add_argument(
        "--pa-ibo-db-values",
        type=float,
        nargs="+",
        default=[0.0, 3.0, 5.0, 6.5, 9.0],
        help="PA input-back-off values in dB.",
    )
    parser.add_argument("--pa-rho", type=float, default=3.0)
    parser.add_argument(
        "--pa-model-version",
        default=PA_RESULT_MODEL_VERSION,
    )
    parser.add_argument(
        "--output-dir",
        default=SCRIPT_DIR / "plots",
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--no-prediction",
        action="store_true",
        help="Exclude CSI prediction curves from the plots.",
    )
    parser.add_argument(
        "--link-adapt",
        default=True,
        action="store_true",
        help="Plot link adaptation results saved when link adaptation was enabled.",
    )

    args = parser.parse_args()

    base_dir = _resolve_path(args.base_dir, SCRIPT_DIR)
    output_dir = _resolve_path(args.output_dir, SCRIPT_DIR)

    scenarios = _default_scenarios(
        include_prediction=not args.no_prediction, link_adapt=args.link_adapt
    )

    cfg = PlotConfig(
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
        channelmamba_seen_drops=args.channelmamba_seen_drops,
        channelmamba_all_drops=args.channelmamba_all_drops,
        wesn_lite_readout_mode=args.wesn_lite_readout_mode,
        sync_errors=args.sync_errors,
        sync_frequency_std_ppb=args.sync_frequency_std_ppb,
        sync_initial_timing_std_ps=args.sync_initial_timing_std_ps,
        sync_initial_phase_std_deg=args.sync_initial_phase_std_deg,
        sync_phase_noise_s100_dbchz=args.sync_phase_noise_s100_dbchz,
        feedback_delay_ms=args.feedback_delay_ms,
        pa_rho=args.pa_rho,
        pa_model_version=args.pa_model_version,
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
    channelmamba_scenario = next(
        (s for s in cfg.scenarios if s.prediction_method == "channelmamba"),
        None,
    )
    channelmamba_time_split_agg = (
        aggregate_metrics(
            loader,
            [channelmamba_scenario],
            cfg.rx_ues,
            cfg.tx_ues,
            cfg.modulation_orders,
            cfg.code_rates,
            drops=cfg.channelmamba_all_drops,
        )
        if channelmamba_scenario is not None
        else {}
    )

    # BER vs RUs (fixed Rx)
    tx_display = [tx + 1 for tx in cfg.tx_ues]
    rx_display = [rx + 2 for rx in cfg.rx_ues]

    ber_tx_series = []
    coded_ber_tx_series = []
    for scenario in cfg.scenarios:
        if scenario.prediction_method == "channelmamba":
            continue
        scenario_values = []
        coded_scenario_values = []
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
            coded_scenario_values.append(
                datapoint.coded_ber if datapoint else np.nan
            )
        ber_tx_series.append((scenario.label, scenario_values))
        coded_ber_tx_series.append((scenario.label, coded_scenario_values))
    if channelmamba_scenario is not None:
        scenario_values = []
        coded_scenario_values = []
        for tx in cfg.tx_ues:
            datapoint = _average_metric(
                channelmamba_time_split_agg,
                channelmamba_scenario,
                cfg.fixed_rx_for_tx_sweep,
                tx,
                cfg.ber_modulation_order,
                float(cfg.ber_code_rate),
            )
            scenario_values.append(datapoint.uncoded_ber if datapoint else np.nan)
            coded_scenario_values.append(datapoint.coded_ber if datapoint else np.nan)
        ber_tx_series.append(("ChannelMamba", scenario_values))
        coded_ber_tx_series.append(("ChannelMamba", coded_scenario_values))

    semilogy_metric(
        tx_display,
        ber_tx_series,
        xlabel="Number of RUs",
        ylabel="Uncoded BER",
        title=f"Uncoded BER vs RUs (UEs={cfg.fixed_rx_for_tx_sweep+2}, MCS={cfg.ber_modulation_order}/{cfg.ber_code_rate})", # treating rx BS as 2 UEs
        output_path=os.path.join(cfg.output_dir, "uncoded_ber_vs_tx_ues.png"),
    )

    semilogy_metric(
        tx_display,
        coded_ber_tx_series,
        xlabel="Number of RUs",
        ylabel="Coded BER",
        title=f"Coded BER vs RUs (UEs={cfg.fixed_rx_for_tx_sweep+2}, MCS={cfg.ber_modulation_order}/{cfg.ber_code_rate})",  # treating rx BS as 2 UEs
        output_path=os.path.join(cfg.output_dir, "coded_ber_vs_tx_ues.png"),
    )

    # BER vs UEs (fixed Tx)
    ber_rx_series = []
    coded_ber_rx_series = []
    for scenario in cfg.scenarios:
        if scenario.prediction_method == "channelmamba":
            continue
        scenario_values = []
        coded_scenario_values = []
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
            coded_scenario_values.append(
                datapoint.coded_ber if datapoint else np.nan
            )
        ber_rx_series.append((scenario.label, scenario_values))
        coded_ber_rx_series.append((scenario.label, coded_scenario_values))

    if channelmamba_scenario is not None:
        scenario_values = []
        coded_scenario_values = []
        for rx in cfg.rx_ues:
            datapoint = _average_metric(
                channelmamba_time_split_agg,
                channelmamba_scenario,
                rx,
                cfg.fixed_tx_for_rx_sweep,
                cfg.ber_modulation_order,
                float(cfg.ber_code_rate),
            )
            scenario_values.append(datapoint.uncoded_ber if datapoint else np.nan)
            coded_scenario_values.append(datapoint.coded_ber if datapoint else np.nan)
        ber_rx_series.append(("ChannelMamba", scenario_values))
        coded_ber_rx_series.append(("ChannelMamba", coded_scenario_values))

    semilogy_metric(
        rx_display,
        ber_rx_series,
        xlabel="Number of UEs",
        ylabel="Uncoded BER",
        title=f"Uncoded BER vs UEs (RUs={cfg.fixed_tx_for_rx_sweep+2}, MCS={cfg.ber_modulation_order}/{cfg.ber_code_rate})",  # treating tx BS as 2 UEs
        output_path=os.path.join(cfg.output_dir, "uncoded_ber_vs_rx_ues.png"),
    )

    semilogy_metric(
        rx_display,
        coded_ber_rx_series,
        xlabel="Number of UEs",
        ylabel="Coded BER",
        title=f"Coded BER vs UEs (RUs={cfg.fixed_tx_for_rx_sweep+2}, MCS={cfg.ber_modulation_order}/{cfg.ber_code_rate})",  # treating tx BS as 2 UEs
        output_path=os.path.join(cfg.output_dir, "coded_ber_vs_rx_ues.png"),
    )

    # Throughput vs RUs (fixed Rx, best MCS)
    thr_tx_series = []
    best_mcs_tx = {}
    throughput_title_descriptor = "Link adaptation" if cfg.link_adapt else "best MCS"
    for scenario in cfg.scenarios:
        if scenario.prediction_method == "channelmamba":
            continue
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

    if channelmamba_scenario is not None:
        scenario_thr = []
        for tx in cfg.tx_ues:
            best_throughput, _ = select_best_mcs(
                channelmamba_time_split_agg,
                channelmamba_scenario,
                cfg.fixed_rx_for_tx_sweep,
                tx,
                cfg.modulation_orders,
                cfg.code_rates,
            )
            scenario_thr.append(best_throughput if best_throughput is not None else np.nan)
        thr_tx_series.append(("ChannelMamba", scenario_thr))
    
    plot_metric(
        tx_display,
        thr_tx_series,
        xlabel="Number of RUs",
        ylabel="Throughput (Mbps)",
        title=f"Throughput vs RUs (UEs={cfg.fixed_rx_for_tx_sweep+2}, {throughput_title_descriptor})", # treating rx BS as 2 UEs
        output_path=os.path.join(cfg.output_dir, "throughput_vs_tx_ues.png"),
    )

    # Throughput vs UEs (fixed Tx, best MCS)
    thr_rx_series = []
    best_mcs_rx = {}
    for scenario in cfg.scenarios:
        if scenario.prediction_method == "channelmamba":
            continue
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

    if channelmamba_scenario is not None:
        scenario_thr = []
        for rx in cfg.rx_ues:
            best_throughput, _ = select_best_mcs(
                channelmamba_time_split_agg,
                channelmamba_scenario,
                rx,
                cfg.fixed_tx_for_rx_sweep,
                cfg.modulation_orders,
                cfg.code_rates,
            )
            scenario_thr.append(best_throughput if best_throughput is not None else np.nan)
        thr_rx_series.append(("ChannelMamba", scenario_thr))
    
    plot_metric(
        rx_display,
        thr_rx_series,
        xlabel="Number of UEs",
        ylabel="Throughput (Mbps)",
        title=f"Throughput vs UEs (RUs={cfg.fixed_tx_for_rx_sweep+2}, {throughput_title_descriptor})", # treating tx BS as 2 UEs
        output_path=os.path.join(cfg.output_dir, "throughput_vs_rx_ues.png"),
    )

    if args.plot_pa_sweep:
        pa_scenarios = [
            scenario
            for scenario in cfg.scenarios
            if scenario.prediction
            and scenario.prediction_method != "channelmamba"
        ]
        pa_throughput = pa_metric_statistics(
            ResultLoader,
            cfg,
            pa_scenarios,
            args.pa_ibo_db_values,
            metric="throughput",
        )
        plot_metric(
            args.pa_ibo_db_values,
            [(item.scenario.label, item.means) for item in pa_throughput],
            xlabel="Input back-off (dB)",
            ylabel="Throughput (Mbps)",
            title=(
                "Throughput vs PA input back-off "
                f"(UEs={cfg.fixed_rx_for_tx_sweep + 2}, "
                f"RUs={cfg.fixed_tx_for_rx_sweep + 1})"
            ),
            output_path=os.path.join(
                cfg.output_dir, "throughput_vs_pa_ibo_db.png"
            ),
            series_errors=[
                (item.scenario.label, item.ci95) for item in pa_throughput
            ],
        )

        pa_nmse = pa_metric_statistics(
            ResultLoader,
            cfg,
            pa_scenarios,
            args.pa_ibo_db_values,
            metric="channel_prediction_nmse",
        )
        semilogy_metric(
            args.pa_ibo_db_values,
            [(item.scenario.label, item.means) for item in pa_nmse],
            xlabel="Input back-off (dB)",
            ylabel="Channel-prediction NMSE",
            title=(
                "Channel-prediction NMSE vs PA input back-off "
                f"(UEs={cfg.fixed_rx_for_tx_sweep + 2}, "
                f"RUs={cfg.fixed_tx_for_rx_sweep + 1})"
            ),
            output_path=os.path.join(
                cfg.output_dir, "channel_prediction_nmse_vs_pa_ibo_db.png"
            ),
        )

        pa_ber = pa_metric_statistics(
            ResultLoader,
            cfg,
            pa_scenarios,
            args.pa_ibo_db_values,
            metric="uncoded_ber",
        )
        semilogy_metric(
            args.pa_ibo_db_values,
            [(item.scenario.label, item.means) for item in pa_ber],
            xlabel="Input back-off (dB)",
            ylabel="Uncoded BER",
            title=(
                "Uncoded BER vs PA input back-off "
                f"(UEs={cfg.fixed_rx_for_tx_sweep + 2}, "
                f"RUs={cfg.fixed_tx_for_rx_sweep + 1})"
            ),
            output_path=os.path.join(
                cfg.output_dir, "uncoded_ber_vs_pa_ibo_db.png"
            ),
        )

    sync_scenarios = [
        scenario
        for scenario in cfg.scenarios
        if scenario.prediction_method != "channelmamba"
    ]
    timing_stats = sync_throughput_statistics(
        ResultLoader,
        cfg,
        sync_scenarios,
        args.sync_initial_timing_std_ps_values,
        sweep="timing",
    )
    plot_metric(
        args.sync_initial_timing_std_ps_values,
        [
            (item.scenario.label, item.means) for item in timing_stats
        ],
        xlabel="Initial timing-error standard deviation (ps)",
        ylabel="Throughput (Mbps)",
        title=(
            "Throughput vs initial timing synchronization error "
            f"(other sync errors=0, UEs={cfg.fixed_rx_for_tx_sweep + 2}, "
            f"RUs={cfg.fixed_tx_for_rx_sweep + 1})"
        ),
        output_path=os.path.join(
            cfg.output_dir,
            "throughput_vs_sync_initial_timing_std_ps_"
            f"fb_delay_ms_{_filename_token(cfg.feedback_delay_ms)}.png",
        ),
        series_errors=[
            (item.scenario.label, item.ci95) for item in timing_stats
        ],
    )
    frequency_stats = sync_throughput_statistics(
        ResultLoader,
        cfg,
        sync_scenarios,
        args.sync_frequency_std_ppb_values,
        sweep="frequency",
    )
    plot_metric(
        args.sync_frequency_std_ppb_values,
        [
            (item.scenario.label, item.means) for item in frequency_stats
        ],
        xlabel="Residual fractional-frequency standard deviation (ppb)",
        ylabel="Throughput (Mbps)",
        title=(
            "Throughput vs frequency synchronization error "
            f"(other sync errors=0, UEs={cfg.fixed_rx_for_tx_sweep + 2}, "
            f"RUs={cfg.fixed_tx_for_rx_sweep + 1})"
        ),
        output_path=os.path.join(
            cfg.output_dir,
            "throughput_vs_sync_frequency_std_ppb_"
            f"fb_delay_ms_{_filename_token(cfg.feedback_delay_ms)}.png",
        ),
        series_errors=[
            (item.scenario.label, item.ci95) for item in frequency_stats
        ],
    )
    phase_sweep_values: List[Optional[float]] = [
        None,
        *args.sync_phase_noise_s100_dbchz_values,
    ]
    phase_stats = sync_throughput_statistics(
        ResultLoader,
        cfg,
        sync_scenarios,
        phase_sweep_values,
        sweep="phase_noise",
    )
    phase_off_x = min(args.sync_phase_noise_s100_dbchz_values) - 10.0
    phase_x_values = [phase_off_x, *args.sync_phase_noise_s100_dbchz_values]
    phase_tick_labels = [
        "Off",
        *(f"{value:g}" for value in args.sync_phase_noise_s100_dbchz_values),
    ]
    phase_delay_slots = (
        args.sync_phase_innovation_delay_slots
        if args.sync_phase_innovation_delay_slots is not None
        else int(round(args.feedback_delay_ms / args.sync_slot_duration_ms))
    )
    phase_duration_s = phase_delay_slots * args.sync_slot_duration_ms * 1e-3
    phase_innovation_labels = [
        "0",
        *(
            f"{phase_noise_rms_deg(value, duration_s=phase_duration_s):.3g}"
            for value in args.sync_phase_noise_s100_dbchz_values
        ),
    ]
    plot_metric(
        phase_x_values,
        [
            (item.scenario.label, item.means) for item in phase_stats
        ],
        xlabel=r"Phase-noise level $S_{100}$ (dBc/Hz)",
        ylabel="Throughput (Mbps)",
        title=(
            "Throughput vs oscillator phase noise "
            f"(other sync errors=0, UEs={cfg.fixed_rx_for_tx_sweep + 2}, "
            f"RUs={cfg.fixed_tx_for_rx_sweep + 1})"
        ),
        output_path=os.path.join(
            cfg.output_dir,
            "throughput_vs_sync_phase_noise_s100_dbchz_"
            f"fb_delay_ms_{_filename_token(cfg.feedback_delay_ms)}.png",
        ),
        series_errors=[
            (item.scenario.label, item.ci95) for item in phase_stats
        ],
        x_tick_labels=phase_tick_labels,
        top_tick_labels=phase_innovation_labels,
        top_xlabel=(
            "RMS phase innovation over "
            f"{phase_delay_slots}-slot CSI delay (degrees)"
        ),
    )

    # Print the maximizing MCS selections for throughput plots
    if not args.link_adapt:
        print("\nMaximizing MCS for Throughput vs RUs (UEs fixed at {}):".format(cfg.fixed_rx_for_tx_sweep+2)) # treating rx BS as 2 UEs
        for scenario in cfg.scenarios:
            print(f"  Scenario: {scenario.label}")
            for tx, mcs in zip(tx_display, best_mcs_tx.get(scenario, [])):
                print(
                    f"    RUs={tx}: {'None' if mcs is None else f'Mod {mcs[0]}, Code rate {mcs[1]}'}"
                )
        print("\nMaximizing MCS for Throughput vs UEs (RUs fixed at {}):".format(cfg.fixed_tx_for_rx_sweep+2)) # treating tx BS as 2 UEs
        for scenario in cfg.scenarios:
            print(f"  Scenario: {scenario.label}")
            for rx, mcs in zip(rx_display, best_mcs_rx.get(scenario, [])):
                print(
                    f"    UEs={rx}: {'None' if mcs is None else f'Mod {mcs[0]}, Code rate {mcs[1]}'}"
                )

if __name__ == "__main__":
    main()
