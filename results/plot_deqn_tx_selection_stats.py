"""Plot DEQN reward/throughput trends for MU-MIMO Tx selection runs.

This script targets the DEQN-based Tx selector used by
``sims/sim_mu_mimo_tx_selection.py``. It loads reward logs saved as
``deqn_tx_rewards_drop_<drop>_rx_UE_<rx>_tx_UE_<tx>_tx_selection_<method>.npz``
then plots the per-step reward trend across drops. Throughput is loaded from
``mu_mimo_results_*_rx_UE_<rx>_tx_UE_<tx>_*_tx_selection_<method>.npz`` files and
aggregated across drops to plot per-step throughput.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DROPS = list(range(1, 101))
DEFAULT_MOBILITY = "high_mobility"
DEFAULT_RX_UES = 4
DEFAULT_TX_UES = 2
DEFAULT_LINK_ADAPT = False
DEFAULT_PERFECT_CSI = True
DEFAULT_CSI_PREDICTION = False
DEFAULT_CHANNEL_PREDICTION_SETTING = "none"
DEFAULT_TX_SELECTION_METHOD = "rl_tx"
DEFAULT_ROLLING_WINDOW_LEN = 100

REWARD_PATTERN = re.compile(
    r"deqn_tx_rewards_drop_(\d+)_rx_UE_(\d+)_tx_UE_(\d+)_"
    r"tx_selection_([A-Za-z0-9_]+)\.npz$"
)

THROUGHPUT_PATTERN = re.compile(
    r"mu_mimo_results_(?P<mcs>.+?)_rx_UE_(\d+)_tx_UE_(\d+)_"
    r"(?P<prediction>prediction_[A-Za-z0-9_]+|perfect_CSI_(True|False))_"
    r"pmi_quantization_(?P<quant>True|False)_tx_selection_(?P<selection>[A-Za-z0-9_]+)\.npz$"
)


@dataclass(frozen=True)
class RewardFile:
    path: Path
    drop_id: int
    rx_ue: int
    tx_ue: int
    selection_method: str


@dataclass(frozen=True)
class ThroughputFile:
    path: Path
    drop_id: int
    rx_ue: int
    tx_ue: int
    prediction_mode: str
    mcs: str
    selection_method: str


def _extract_reward_metadata(path: Path) -> RewardFile:
    match = REWARD_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse reward metadata from {path}")
    return RewardFile(
        path=path,
        drop_id=int(match.group(1)),
        rx_ue=int(match.group(2)),
        tx_ue=int(match.group(3)),
        selection_method=match.group(4),
    )


def _extract_throughput_metadata(path: Path, drop_id: int) -> ThroughputFile:
    match = THROUGHPUT_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse throughput metadata from {path}")

    return ThroughputFile(
        path=path,
        drop_id=drop_id,
        rx_ue=int(match.group(2)),
        tx_ue=int(match.group(3)),
        prediction_mode=match.group("prediction"),
        mcs=match.group("mcs"),
        selection_method=match.group("selection"),
    )


def _find_reward_files(
    root: Path,
    drops: Iterable[int],
    mobility: str,
    rx_ue: int,
    tx_ue: int,
    selection_method: str,
) -> List[RewardFile]:
    files: List[RewardFile] = []
    for drop in drops:
        drop_path = root / f"channels_{mobility}_{drop}"
        if not drop_path.exists():
            print(f"Warning: Drop directory not found: {drop_path}")
            continue

        candidates: List[RewardFile] = []
        for path in sorted(drop_path.glob("deqn_tx_rewards_drop_*_rx_UE_*_tx_UE_*_tx_selection_*.npz")):
            try:
                info = _extract_reward_metadata(path)
            except ValueError:
                continue
            if info.drop_id != int(drop):
                continue
            if info.rx_ue != rx_ue or info.tx_ue != tx_ue:
                continue
            if info.selection_method != selection_method:
                continue
            candidates.append(info)

        if not candidates:
            print(f"Warning: No reward file found for drop {drop} under {drop_path}")
            continue

        files.extend(candidates)

    return files


def _find_throughput_files(
    root: Path,
    drops: Iterable[int],
    mobility: str,
    rx_ue: int,
    tx_ue: int,
    selection_method: str,
    prediction_mode: str,
) -> List[ThroughputFile]:
    files: List[ThroughputFile] = []
    for drop in drops:
        drop_path = root / f"channels_{mobility}_{drop}"
        if not drop_path.exists():
            print(f"Warning: Drop directory not found: {drop_path}")
            continue

        candidates: List[ThroughputFile] = []
        for path in sorted(drop_path.glob("mu_mimo_results_*_rx_UE_*_tx_UE_*_tx_selection_*.npz")):
            try:
                info = _extract_throughput_metadata(path, drop_id=int(drop))
            except ValueError:
                continue
            if info.rx_ue != rx_ue or info.tx_ue != tx_ue:
                continue
            if info.selection_method != selection_method:
                continue
            if info.prediction_mode != prediction_mode:
                continue
            candidates.append(info)

        if not candidates:
            print(
                "Warning: Throughput file not found for drop "
                f"{drop}: expected rx {rx_ue}, tx {tx_ue} under {drop_path}"
            )
            continue

        files.extend(candidates)

    return files


def _load_rewards(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    rewards = np.asarray(data["rewards"], dtype=float)

    if rewards.ndim == 1:
        return rewards
    if rewards.ndim == 2 and rewards.shape[1] >= 1:
        return rewards[:, -1]
    raise ValueError(f"Unexpected reward array shape {rewards.shape} in {path}")


def _load_throughput(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "per_step_throughput" not in data:
        raise ValueError(f"Per-step throughput array missing from {path}")

    per_step_throughput = np.asarray(data["per_step_throughput"], dtype=float).ravel()
    if per_step_throughput.size == 0:
        raise ValueError(f"Empty per-step throughput array in {path}")

    return per_step_throughput


def _aggregate_rewards(files: Iterable[RewardFile]) -> Tuple[List[Tuple[int, float]], List[int]]:
    ordered_files = sorted(files, key=lambda info: (info.drop_id, info.path.name))
    series: List[Tuple[int, float]] = []
    step_idx = 1

    for file_info in ordered_files:
        rewards = _load_rewards(file_info.path)
        for reward in rewards:
            series.append((step_idx, float(reward)))
            step_idx += 1
    step_ids = [step for step, _ in series]
    return series, step_ids


def _aggregate_throughput(files: Iterable[ThroughputFile]) -> Tuple[List[Tuple[int, float]], List[int]]:
    ordered_files = sorted(files, key=lambda info: (info.drop_id, info.path.name))
    series: List[Tuple[int, float]] = []
    step_idx = 1

    for file_info in ordered_files:
        throughput_values = _load_throughput(file_info.path)
        for value in throughput_values:
            series.append((step_idx, float(value)))
            step_idx += 1
    step_ids = [step for step, _ in series]
    return series, step_ids


def _apply_rolling_mean(series: List[Tuple[int, float]], window: int) -> List[Tuple[int, float]]:
    if window <= 1 or len(series) < window:
        return series
    steps, values = zip(*series)
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")
    new_steps = steps[window - 1 :]
    return list(zip(new_steps, smoothed.tolist()))


def _select_tick_steps(step_ids: Iterable[int], stride: int) -> List[int]:
    if stride <= 1:
        return sorted(set(step_ids))
    unique_steps = sorted(set(step_ids))
    if not unique_steps:
        return []
    return list(range(unique_steps[0], unique_steps[-1] + 1, stride))


def plot_rewards(series: List[Tuple[int, float]], step_ids: List[int], output: Path, tick_stride: int) -> None:
    if not series:
        raise RuntimeError("No reward data found to plot.")

    steps, values = zip(*series)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("DEQN Tx-selection rewards across steps")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(_select_tick_steps(step_ids, tick_stride))
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved reward plot to {output}")


def plot_throughput(series: List[Tuple[int, float]], step_ids: List[int], output: Path, tick_stride: int) -> None:
    if not series:
        raise RuntimeError("No throughput data found to plot.")

    steps, values = zip(*series)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, marker="s")
    plt.xlabel("Step")
    plt.ylabel("Per-step throughput")
    plt.title("Tx-selection throughput across steps")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(_select_tick_steps(step_ids, tick_stride))
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved throughput plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DEQN Tx-selection reward/throughput logs across drops.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results") / "channels_multiple_mu_mimo",
        help="Root directory to scan for drop results.",
    )
    parser.add_argument("--drops", type=int, nargs="+", default=DEFAULT_DROPS)
    parser.add_argument("--mobility", type=str, default=DEFAULT_MOBILITY)
    parser.add_argument("--rx-ue", type=int, default=DEFAULT_RX_UES)
    parser.add_argument("--tx-ue", type=int, default=DEFAULT_TX_UES)
    parser.add_argument("--link-adapt", action="store_true", default=DEFAULT_LINK_ADAPT)
    parser.add_argument("--perfect-csi", action="store_true", default=DEFAULT_PERFECT_CSI)
    parser.add_argument("--csi-prediction", action="store_true", default=DEFAULT_CSI_PREDICTION)
    parser.add_argument(
        "--channel-prediction-setting",
        type=str,
        default=DEFAULT_CHANNEL_PREDICTION_SETTING,
    )
    parser.add_argument(
        "--tx-selection-method",
        type=str,
        default=DEFAULT_TX_SELECTION_METHOD,
    )
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW_LEN)
    parser.add_argument(
        "--rewards-output",
        type=Path,
        default=Path("results") / "deqn_tx_rewards.png",
        help="Path to save the generated reward plot image.",
    )
    parser.add_argument(
        "--throughput-output",
        type=Path,
        default=Path("results") / "deqn_tx_throughput.png",
        help="Path to save the generated throughput plot image.",
    )

    args = parser.parse_args()

    if args.perfect_csi:
        prediction_mode = f"perfect_CSI_{args.perfect_csi}"
    elif args.csi_prediction:
        prediction_mode = f"prediction_{args.channel_prediction_setting}"
    else:
        prediction_mode = f"perfect_CSI_{args.perfect_csi}"

    drops = list(args.drops)
    root = args.root

    reward_files = _find_reward_files(
        root,
        drops,
        args.mobility,
        args.rx_ue,
        args.tx_ue,
        args.tx_selection_method,
    )
    throughput_files = _find_throughput_files(
        root,
        drops,
        args.mobility,
        args.rx_ue,
        args.tx_ue,
        args.tx_selection_method,
        prediction_mode,
    )

    if not reward_files and not throughput_files:
        raise SystemExit("No DEQN Tx-selection outputs found for the requested settings.")

    if reward_files:
        reward_series, reward_steps = _aggregate_rewards(reward_files)
        reward_series = _apply_rolling_mean(reward_series, args.rolling_window)
        plot_rewards(reward_series, reward_steps, args.rewards_output, args.rolling_window)
    else:
        print("No reward files found; skipping reward plot.")

    if throughput_files:
        throughput_series, throughput_steps = _aggregate_throughput(throughput_files)
        throughput_series = _apply_rolling_mean(throughput_series, args.rolling_window)
        plot_throughput(throughput_series, throughput_steps, args.throughput_output, args.rolling_window)
    else:
        print("No throughput files found; skipping throughput plot.")


if __name__ == "__main__":
    main()