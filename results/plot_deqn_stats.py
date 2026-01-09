"""Plot single-agent DEQN reward/action/throughput trends for MU-MIMO runs.

This script targets the single-agent TX-side RL strategy used by
``sims/sim_mu_mimo_rl_chan_pred_training_v2.py`` and
``sims/sim_mu_mimo_testing_updates.py``. It loads DEQN rewards/actions saved as
``deqn_rewards_drop_<drop>_rx_UE_<rx>_tx_UE_<tx>_imitation_<method>_steps_<n>.npz``
and ``deqn_actions_drop_<drop>_rx_UE_<rx>_tx_UE_<tx>_imitation_<method>_steps_<n>.npz``
then plots per-drop average reward plus step-wise action selections.

Throughput is loaded from files named like:
``mu_mimo_results_<mcs>_rx_UE_<rx>_tx_UE_<tx>_prediction_<method>_pmi_quantization_<bool>_imitation_<method>_steps_<n>.npz``.

"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DROPS = list(range(4, 45))
DEFAULT_MOBILITY = "high_mobility"
DEFAULT_RX_UES = 4
DEFAULT_TX_UES = 6
DEFAULT_LINK_ADAPT = True
DEFAULT_PERFECT_CSI = False
DEFAULT_CSI_PREDICTION = True
DEFAULT_CHANNEL_PREDICTION_SETTING = "deqn_plus_two_mode"
DEFAULT_IMITATION_METHOD = "none"
DEFAULT_IMITATION_DROP_COUNT = 0
DEFAULT_ROLLING_WINDOW_LEN = 20

REWARD_PATTERN = re.compile(
    r"deqn_rewards_drop_(\d+)_rx_UE_(\d+)_tx_UE_(\d+)_"
    r"imitation_([A-Za-z0-9_]+)_steps_(\d+)\.npz$"
)

ACTION_PATTERN = re.compile(
    r"deqn_actions_drop_(\d+)_rx_UE_(\d+)_tx_UE_(\d+)_"
    r"imitation_([A-Za-z0-9_]+)_steps_(\d+)\.npz$"
)


THROUGHPUT_PATTERN = re.compile(
    r"mu_mimo_results_(?P<mcs>.+?)_rx_UE_(\d+)_tx_UE_(\d+)_"
    r"prediction_(?P<prediction>[A-Za-z0-9_]+)_pmi_quantization_(?P<quant>True|False)_"
    r"imitation_([A-Za-z0-9_]+)_steps_(\d+)\.npz$"
)


@dataclass(frozen=True)
class RewardFile:
    path: Path
    drop_id: int
    rx_ue: int
    tx_ue: int
    imitation_method: str
    imitation_steps: int
    

@dataclass(frozen=True)
class ActionFile:
    path: Path
    drop_id: int
    rx_ue: int
    tx_ue: int
    imitation_method: str
    imitation_steps: int

@dataclass(frozen=True)
class ThroughputFile:
    path: Path
    drop_id: int
    rx_ue: int
    tx_ue: int
    prediction_method: str
    mcs: str
    imitation_method: str
    imitation_steps: int


def _extract_reward_metadata(path: Path) -> RewardFile:
    match = REWARD_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse action metadata from {path}")
    return RewardFile(
        path=path,
        drop_id=int(match.group(1)),
        rx_ue=int(match.group(2)),
        tx_ue=int(match.group(3)),
        imitation_method=match.group(4),
        imitation_steps=int(match.group(5)),
    )

def _extract_action_metadata(path: Path) -> ActionFile:
    match = ACTION_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse reward metadata from {path}")
    return ActionFile(
        path=path,
        drop_id=int(match.group(1)),
        rx_ue=int(match.group(2)),
        tx_ue=int(match.group(3)),
        imitation_method=match.group(4),
        imitation_steps=int(match.group(5)),
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
        prediction_method=match.group("prediction"),
        mcs=match.group("mcs"),
        imitation_method=match.group(6),
        imitation_steps=int(match.group(7)),
    )



def _find_reward_files(
    root: Path, drops: Iterable[int], mobility: str, rx_ue: int, tx_ue: int
) -> List[RewardFile]:

    files: List[RewardFile] = []
    for drop in drops:
        drop_path = root / f"channels_{mobility}_{drop}"
        if not drop_path.exists():
            print(f"Warning: Drop directory not found: {drop_path}")
            continue

        candidates: List[RewardFile] = []
        for path in sorted(drop_path.glob("deqn_rewards_drop_*_rx_UE_*_tx_UE_*_imitation_*_steps_*.npz")):
            try:
                info = _extract_reward_metadata(path)
            except ValueError:
                continue
            if info.drop_id != int(drop):
                continue
            if info.rx_ue != rx_ue or info.tx_ue != tx_ue:
                continue

            candidates.append(info)

        if not candidates:
            print(f"Warning: No reward file found for drop {drop} under {drop_path}")
            continue

        files.extend(candidates)

    return files

def _find_action_files(
    root: Path, drops: Iterable[int], mobility: str, rx_ue: int, tx_ue: int
) -> List[ActionFile]:

    files: List[ActionFile] = []
    for drop in drops:
        drop_path = root / f"channels_{mobility}_{drop}"
        if not drop_path.exists():
            print(f"Warning: Drop directory not found: {drop_path}")
            continue

        candidates: List[ActionFile] = []
        for path in sorted(drop_path.glob("deqn_actions_drop_*_rx_UE_*_tx_UE_*_imitation_*_steps_*.npz")):
            try:
                info = _extract_action_metadata(path)
            except ValueError:
                continue
            if info.drop_id != int(drop):
                continue
            if info.rx_ue != rx_ue or info.tx_ue != tx_ue:
                continue

            candidates.append(info)

        if not candidates:
            print(f"Warning: No action file found for drop {drop} under {drop_path}")
            continue

        files.extend(candidates)

    return files

def _find_throughput_files(
    root: Path, drops: Iterable[int], mobility: str, rx_ue: int, tx_ue: int, prediction_method: str
) -> List[ThroughputFile]:

    files: List[ThroughputFile] = []
    for drop in drops:
        drop_path = root / f"channels_{mobility}_{drop}"
        if not drop_path.exists():
            print(f"Warning: Drop directory not found: {drop_path}")
            continue

        candidates: List[ThroughputFile] = []
        for path in sorted(
            drop_path.glob("mu_mimo_results_*_rx_UE_*_tx_UE_*_prediction_*_pmi_quantization_*_imitation_*_steps_*.npz")
        ):
            try:
                info = _extract_throughput_metadata(path, drop_id=int(drop))
            except ValueError:
                continue

            if info.rx_ue != rx_ue or info.tx_ue != tx_ue:
                continue
            if info.prediction_method != prediction_method:
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

    if rewards.ndim == 2:
        rewards = rewards[:,-1]
    else:
        raise ValueError(f"Unexpected reward array shape {rewards.shape} in {path}")
    return rewards

def _load_actions(path: Path) -> np.ndarray:

    data = np.load(path, allow_pickle=False)
    actions = np.asarray(data["actions"], dtype=int)

    actions_out = np.zeros([actions.shape[0], 2])
    if actions.ndim == 2:
        actions_out[:, 0] = actions[:, 0]
        actions_out[:, 1] = actions[:, -1]
    else:
        raise ValueError(f"Unexpected action array shape {actions.shape} in {path}")

    return actions_out 

def _load_throughput(path: Path) -> float:

    data = np.load(path, allow_pickle=True)
    if "throughput" not in data:
        raise ValueError(f"Throughput array missing from {path}")

    throughput = np.asarray(data["throughput"], dtype=float).ravel()
    if throughput.size == 0:
        raise ValueError(f"Empty throughput array in {path}")

    return float(np.nanmean(throughput))

def _aggregate_rewards(files: Iterable[RewardFile]) -> Tuple[List[Tuple[int, float]], List[int]]:
    by_drop: dict[int, List[float]] = {}

    for file_info in files:
        rewards = _load_rewards(file_info.path)
        drop_rewards = by_drop.setdefault(file_info.drop_id, [])
        for reward in rewards:
            drop_rewards.append(float(reward))
    series = [(drop, float(np.mean(values))) for drop, values in by_drop.items()]
    series.sort(key=lambda item: item[0])
    drop_ids = [drop for drop, _ in series]
    return series, drop_ids

def _aggregate_actions(files: Iterable[ActionFile]) -> Tuple[List[Tuple[int, int]], int]:
    actions_series: List[Tuple[int, int]] = []
    max_step = 0
    for file_info in files:
        actions = _load_actions(file_info.path)
        for step, action_idx in actions:
            actions_series.append((int(step), int(action_idx)))
            max_step = max(max_step, int(step))
    actions_series.sort(key=lambda item: item[0])
    return actions_series, max_step

def _aggregate_throughput(files: Iterable[ThroughputFile]) -> Tuple[List[Tuple[int, float]], List[int]]:
    series = [(file_info.drop_id, _load_throughput(file_info.path)) for file_info in files]
    series.sort(key=lambda item: item[0])
    drop_ids = [drop for drop, _ in series]
    return series, drop_ids

def _apply_rolling_mean(series: List[Tuple[int, float]], window: int) -> List[Tuple[int, float]]:
    if window <= 1 or len(series) < window:
        return series
    drops, values = zip(*series)
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")
    new_drops = drops[window - 1 :]
    return list(zip(new_drops, smoothed.tolist()))

def plot_rewards(series: List[Tuple[int, float]], drop_ids: List[int], output: Path) -> None:
    if not series:
        raise RuntimeError("No reward data found to plot.")

    drops, values = zip(*series)

    plt.figure(figsize=(10, 6))
    plt.plot(drops, values, marker="o")
    plt.xlabel("Drop")
    plt.ylabel("Average reward")
    plt.title("DEQN average reward across drops")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(drop_ids)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved reward plot to {output}")

def plot_actions(series: List[Tuple[int, int]], max_step: int, output: Path) -> None:
    if not series:
        raise RuntimeError("No action data found to plot.")
    steps, actions = zip(*series)

    plt.figure(figsize=(10, 6))
    plt.step(steps, actions, where="post")
    plt.xlabel("Step")
    plt.ylabel("Action index")
    plt.title("DEQN actions across steps")
    if max_step > 0:
        plt.xlim(1, max_step)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved action plot to {output}")

def plot_throughput(series: List[Tuple[int, float]], drop_ids: List[int], output: Path) -> None:
    if not series:
        raise RuntimeError("No throughput data found to plot.")

    drops, values = zip(*series)
    plt.figure(figsize=(10, 6))

    plt.plot(drops, values, marker="s")
    plt.xlabel("Drop")
    plt.ylabel("Average throughput")
    plt.title("DEQN throughput across drops")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(drop_ids)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved throughput plot to {output}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DEQN reward/action logs across drops.")
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

    parser.add_argument("--imitation-method", type=str, default=DEFAULT_IMITATION_METHOD)
    parser.add_argument("--imitation-drop-count", type=int, default=DEFAULT_IMITATION_DROP_COUNT)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW_LEN)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "deqn_rewards.png",
        help="Path to save the generated reward plot image.",
    )

    parser.add_argument(
        "--actions-output",
        type=Path,
        default=Path("results") / "deqn_actions.png",
        help="Path to save the generated action plot image.",
    )

    parser.add_argument(
        "--throughput-output",
        type=Path,
        default=Path("results") / "deqn_throughput.png",
        help="Path to save the generated throughput plot image.",
    )

    args = parser.parse_args()

    if args.perfect_csi:
        prediction_method = "perfect_csi"
    elif args.csi_prediction:
        prediction_method = args.channel_prediction_setting
    else:
        prediction_method = "none"

    drops = list(args.drops)
    root = args.root

    reward_files = _find_reward_files(root, drops, args.mobility, args.rx_ue, args.tx_ue)
    action_files = _find_action_files(root, drops, args.mobility, args.rx_ue, args.tx_ue)
    throughput_files = _find_throughput_files(
        root, drops, args.mobility, args.rx_ue, args.tx_ue, prediction_method
    )

    if not reward_files and not action_files and not throughput_files:
        raise SystemExit("No DEQN outputs found for the requested settings.")

    if reward_files:
        reward_series, reward_drops = _aggregate_rewards(reward_files)
        reward_series = _apply_rolling_mean(reward_series, args.rolling_window)
        plot_rewards(reward_series, reward_drops, args.output)
    else:
        print("No reward files found; skipping reward plot.")

    if action_files:
        action_series, max_step = _aggregate_actions(action_files)
        plot_actions(action_series, max_step, args.actions_output)
    else:
        print("No action files found; skipping action plot.")

    if throughput_files:
        throughput_series, throughput_drops = _aggregate_throughput(throughput_files)
        throughput_series = _apply_rolling_mean(throughput_series, args.rolling_window)
        plot_throughput(throughput_series, throughput_drops, args.throughput_output)
    else:
        print("No throughput files found; skipping throughput plot.")

if __name__ == "__main__":
    main()