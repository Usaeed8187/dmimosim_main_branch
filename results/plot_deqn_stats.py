"""Plot DEQN reward and throughput trends across drops and agent pairs.

This utility scans the results directory for files named
``deqn_rewards_drop_<drop>[_rx_UE_<rx>_tx_UE_<tx>].npz`` and
``deqn_actions_drop_<drop>[_rx_UE_<rx>_tx_UE_<tx>].npz`` (produced by
``sims/sim_mu_mimo_deqn_chan_pred_training.py``) and generates line plots
showing the average reward per (rx, tx) agent pair for each drop, as well as
the action indices chosen at every time step.

Throughput is loaded from files named
``mu_mimo_results_link_adapt_rx_UE_<rx>_tx_UE_<tx>_prediction_deqn_pmi_quantization_True.npz``
that live directly inside per-drop folders explicitly supplied on the
command line.

"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


REWARD_PATTERN = re.compile(
    r"deqn_rewards_drop_(\d+)(?:_rx_UE_(\d+)_tx_UE_(\d+))?\.npz$"
)

ACTION_PATTERN = re.compile(
    r"deqn_actions_drop_(\d+)(?:_rx_UE_(\d+)_tx_UE_(\d+))?\.npz$"
)

THROUGHPUT_PATTERN = re.compile(
    r"mu_mimo_results_link_adapt_rx_UE_(\d+)_tx_UE_(\d+)_prediction_deqn_pmi_quantization_True\.npz$"
)

@dataclass(frozen=True)
class RewardFile:
    path: Path
    drop_id: int
    rx_ue: Optional[int]
    tx_ue: Optional[int]

@dataclass(frozen=True)
class ActionFile:
    path: Path
    drop_id: int
    rx_ue: Optional[int]
    tx_ue: Optional[int]

@dataclass(frozen=True)
class ThroughputFile:
    path: Path
    drop_id: int
    rx_ue: int
    tx_ue: int


def _extract_metadata(path: Path) -> RewardFile:
    """Extract drop/rx/tx identifiers from a reward file name."""


    match = REWARD_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse metadata from {path}")
    drop_id = int(match.group(1))
    rx_ue = int(match.group(2)) if match.group(2) else None
    tx_ue = int(match.group(3)) if match.group(3) else None
    return RewardFile(path=path, drop_id=drop_id, rx_ue=rx_ue, tx_ue=tx_ue)

def _extract_action_metadata(path: Path) -> ActionFile:
    """Extract drop/rx/tx identifiers from an action log file name."""

    match = ACTION_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse metadata from {path}")
    drop_id = int(match.group(1))
    rx_ue = int(match.group(2)) if match.group(2) else None
    tx_ue = int(match.group(3)) if match.group(3) else None
    return ActionFile(path=path, drop_id=drop_id, rx_ue=rx_ue, tx_ue=tx_ue)

def _extract_throughput_metadata(path: Path, drop_id: int) -> ThroughputFile:
    """Extract rx/tx identifiers from a throughput result filename."""

    match = THROUGHPUT_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse throughput metadata from {path}")

    rx_ue = int(match.group(1))
    tx_ue = int(match.group(2))
    return ThroughputFile(path=path, drop_id=drop_id, rx_ue=rx_ue, tx_ue=tx_ue)



def _find_reward_files(
    root: Path, drops: Iterable[int], rx_ue: int | None, tx_ue: int | None
) -> List[RewardFile]:

    """Return one DEQN reward file per drop under ``root`` matching the filters."""

    files: List[RewardFile] = []
    for drop in drops:
        drop_path = root / f"channels_high_mobility_{drop}"
        if not drop_path.exists():
            print(f"Warning: Drop directory not found: {drop_path}")
            continue

        candidates: List[RewardFile] = []
        for path in sorted(drop_path.glob("deqn_rewards_drop_*.npz")):
            info = _extract_metadata(path)

            if info.drop_id != int(drop):
                continue
            if rx_ue is not None and info.rx_ue != rx_ue:
                continue
            if tx_ue is not None and info.tx_ue != tx_ue:
                continue

            candidates.append(info)

        if not candidates:
            print(f"Warning: No reward file found for drop {drop} under {drop_path}")
            continue

        if len(candidates) > 1:
            print(
                "Warning: Multiple reward files found for drop "
                f"{drop}; using {candidates[0].path}"
            )

        files.append(candidates[0])

    return files

def _find_action_files(
    root: Path, drops: Iterable[int], rx_ue: int | None, tx_ue: int | None
) -> List[ActionFile]:
    """Return one DEQN action log per drop under ``root`` matching the filters."""

    files: List[ActionFile] = []
    for drop in drops:
        drop_path = root / f"channels_high_mobility_{drop}"
        if not drop_path.exists():
            print(f"Warning: Drop directory not found: {drop_path}")
            continue

        candidates: List[ActionFile] = []
        for path in sorted(drop_path.glob("deqn_actions_drop_*.npz")):
            info = _extract_action_metadata(path)

            if info.drop_id != int(drop):
                continue
            if rx_ue is not None and info.rx_ue != rx_ue:
                continue
            if tx_ue is not None and info.tx_ue != tx_ue:
                continue

            candidates.append(info)

        if not candidates:
            print(f"Warning: No action file found for drop {drop} under {drop_path}")
            continue

        if len(candidates) > 1:
            print(
                "Warning: Multiple action files found for drop "
                f"{drop}; using {candidates[0].path}"
            )

        files.append(candidates[0])

    return files

def _find_throughput_files(
    root: Path, drops: Iterable[int], rx_ue: int, tx_ue: int
) -> List[ThroughputFile]:
    """Return throughput result files under ``root`` for specified drops."""

    files: List[ThroughputFile] = []
    for drop in drops:
        drop_path = root / f"channels_high_mobility_{drop}"
        file_path = drop_path / (
            f"mu_mimo_results_link_adapt_rx_UE_{rx_ue}_tx_UE_{tx_ue}_prediction_"
            "deqn_pmi_quantization_True.npz"
        )


        if not file_path.exists():
            print(f"Warning: Throughput file not found for drop {drop}: {file_path}")
            continue
        info = _extract_throughput_metadata(file_path, drop_id=int(drop))

        files.append(info)

    return files


def _load_rewards(path: Path) -> np.ndarray:
    """Load reward log entries from a NPZ file.

    The stored format is expected to be an array with columns
    ``[rx_idx, tx_idx, reward]``.
    """

    data = np.load(path, allow_pickle=False)
    rewards = np.asarray(data["rewards"], dtype=float)

    if rewards.ndim == 1 and rewards.size % 3 == 0:
        rewards = rewards.reshape(-1, 3)

    if rewards.ndim != 2 or rewards.shape[1] < 3:
        raise ValueError(f"Unexpected reward array shape {rewards.shape} in {path}")

    return rewards[:, :3]

def _load_actions(path: Path) -> np.ndarray:
    """Load action log entries from a NPZ file.

    The stored format is expected to be an array with columns
    ``[step, rx_idx, tx_idx, action_idx]``.
    """

    data = np.load(path, allow_pickle=False)
    actions = np.asarray(data["actions"], dtype=int)

    if actions.ndim == 1 and actions.size % 4 == 0:
        actions = actions.reshape(-1, 4)

    if actions.ndim != 2 or actions.shape[1] < 4:
        raise ValueError(f"Unexpected action array shape {actions.shape} in {path}")

    return actions[:, :4]

def _load_imitation_label(paths: Iterable[Path]) -> Optional[str]:
    """Return the first imitation info string found in the provided NPZ files."""

    for path in paths:
        try:
            data = np.load(path, allow_pickle=True)
        except Exception:
            continue

        if "imitation_info" in data:
            try:
                value = data["imitation_info"]
                return str(value.item()) if hasattr(value, "item") else str(value)
            except Exception:
                continue

    return None

def _aggregate_actions(
    files: Iterable[ActionFile],
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int]]], int]:
    """Collate ordered action selections for each (rx, tx) pair.

    Returns:
        pair_actions: mapping from (rx_idx, tx_idx) to a list of (step, action_idx)
        max_step: highest step index observed across all agents
    """

    pair_actions: DefaultDict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    max_step = 0

    for file_info in files:
        actions = _load_actions(file_info.path)

        for step, rx_idx, tx_idx, action_idx in actions:
            pair_actions[(int(rx_idx), int(tx_idx))].append((int(step), int(action_idx)))
            max_step = max(max_step, int(step))

    for series in pair_actions.values():
        series.sort(key=lambda item: item[0])

    return pair_actions, max_step

def _aggregate_rewards(
    files: Iterable[RewardFile],
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, float]]], List[int]]:
    """Compute per-drop average rewards for each (rx, tx) pair.

    Returns:
        pair_rewards: mapping from (rx_idx, tx_idx) to a list of (drop, mean_reward)
        drop_ids: sorted list of all drop identifiers encountered
    """

    pair_rewards: DefaultDict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
    drop_ids: List[int] = []

    for file_info in files:
        drop_id = file_info.drop_id
        rewards = _load_rewards(file_info.path)

        drop_ids.append(drop_id)
        pair_to_values: DefaultDict[Tuple[int, int], List[float]] = defaultdict(list)

        for rx_idx, tx_idx, reward in rewards:
            pair_to_values[(int(rx_idx), int(tx_idx))].append(float(reward))

        for pair, values in pair_to_values.items():
            pair_rewards[pair].append((drop_id, float(np.mean(values))))

    drop_ids = sorted(set(drop_ids))
    # Sort each pair's series by drop id for consistent plotting
    for series in pair_rewards.values():
        series.sort(key=lambda item: item[0])

    return pair_rewards, drop_ids

def _load_throughput(path: Path) -> float:
    """Load and average throughput values from a simulation result file."""

    data = np.load(path, allow_pickle=True)
    if "throughput" not in data:
        raise ValueError(f"Throughput array missing from {path}")

    throughput = np.asarray(data["throughput"], dtype=float)
    throughput = throughput.ravel()
    if throughput.size == 0:
        raise ValueError(f"Empty throughput array in {path}")

    return float(np.nanmean(throughput))

def plot_actions(
    pair_actions: Dict[Tuple[int, int], List[Tuple[int, int]]],
    max_step: int,
    output: Path,
    show: bool = False,
    imitation_label: Optional[str] = None,
) -> None:
    """Plot chosen action indices per step for each (rx, tx) pair."""

    if not pair_actions:
        raise RuntimeError("No action data found to plot.")

    plt.figure(figsize=(10, 6))
    for (rx_idx, tx_idx), series in sorted(pair_actions.items()):
        steps, actions = zip(*series)
        plt.step(steps, actions, where="post", label=f"rx{rx_idx}-tx{tx_idx}")

    plt.xlabel("Step")
    plt.ylabel("Action index")
    title = "DEQN per-agent actions across steps"
    if imitation_label:
        title += f"\n{imitation_label}"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    if max_step > 0:
        plt.xlim(1, max_step)
    plt.legend()
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved action plot to {output}")

def _aggregate_throughput(
    files: Iterable[ThroughputFile],
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, float]]], List[int]]:
    """Compute per-drop average throughput for each (rx, tx) pair."""

    pair_throughput: DefaultDict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
    drop_ids: List[int] = []

    for file_info in files:
        drop_id = file_info.drop_id
        throughput_value = _load_throughput(file_info.path)

        pair_throughput[(file_info.rx_ue, file_info.tx_ue)].append(
            (drop_id, throughput_value)
        )
        drop_ids.append(drop_id)

    drop_ids = sorted(set(drop_ids))
    for series in pair_throughput.values():
        series.sort(key=lambda item: item[0])

    return pair_throughput, drop_ids

def _apply_rolling_mean(
    pair_series: Dict[Tuple[int, int], List[Tuple[int, float]]], window: int
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, float]]], List[int]]:
    """Apply a trailing rolling mean across drop-series for each pair.

    The resulting drop id for each averaged point corresponds to the end of the
    window (i.e., drop ``window - 1`` uses drops ``0..window-1`` from the
    original sequence).
    """

    if window <= 1:
        return pair_series, sorted({drop for series in pair_series.values() for drop, _ in series})

    rolled: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
    all_drops: List[int] = []

    for pair, series in pair_series.items():
        drops, values = zip(*series)

        if len(series) < window:
            rolled[pair] = list(series)
            all_drops.extend(drops)
            continue

        kernel = np.ones(window, dtype=float) / float(window)
        smoothed = np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")
        new_drops = drops[window - 1 :]
        rolled_series = list(zip(new_drops, smoothed.tolist()))
        rolled[pair] = rolled_series
        all_drops.extend(new_drops)

    return rolled, sorted(set(all_drops))


def plot_rewards(
    pair_rewards: Dict[Tuple[int, int], List[Tuple[int, float]]],
    drop_ids: List[int],
    output: Path,
    show: bool = False,
    imitation_label: Optional[str] = None,
) -> None:
    """Plot mean rewards per (rx, tx) pair across drops."""

    if not pair_rewards:
        raise RuntimeError("No reward data found to plot.")

    plt.figure(figsize=(10, 6))
    for (rx_idx, tx_idx), series in sorted(pair_rewards.items()):
        drops, values = zip(*series)
        plt.plot(drops, values, marker="o", label=f"rx{rx_idx}-tx{tx_idx}")

    plt.xlabel("Drop")
    plt.ylabel("Average reward")
    title = "DEQN per-agent rewards across drops"
    if imitation_label:
        title += f"\n{imitation_label}"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(drop_ids)
    plt.legend()
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved reward plot to {output}")

def plot_throughput(
    pair_throughput: Dict[Tuple[int, int], List[Tuple[int, float]]],
    drop_ids: List[int],
    output: Path,
    show: bool = False,
    imitation_label: Optional[str] = None,
) -> None:
    """Plot mean throughput per (rx, tx) pair across drops."""

    if not pair_throughput:
        raise RuntimeError("No throughput data found to plot.")

    plt.figure(figsize=(10, 6))
    for (rx_idx, tx_idx), series in sorted(pair_throughput.items()):
        drops, values = zip(*series)
        plt.plot(drops, values, marker="s", label=f"rx{rx_idx}-tx{tx_idx}")

    plt.xlabel("Drop")
    plt.ylabel("Average throughput")
    title = "DEQN per-agent throughput across drops"
    if imitation_label:
        title += f"\n{imitation_label}"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(drop_ids)
    plt.legend()
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved throughput plot to {output}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DEQN reward logs across drops.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results") / "channels_multiple_mu_mimo",
        help="Root directory to scan for deqn_rewards_drop_*.npz files.",
    )
    parser.add_argument(
        "--drops",
        type=int,
        nargs="+",
        # default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        default=list(range(1, 33)),
        help="List of drop identifiers to include in the plots.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "deqn_rewards.png",
        help="Path to save the generated plot image.",
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

    parser.add_argument(
        "--rx-ue",
        type=int,
        default=2,
        help="RX UE selection to load reward and throughput files for.",
    )
    parser.add_argument(
        "--tx-ue",
        type=int,
        default=2,
        help="TX UE selection to load reward and throughput files for.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=1,
        help=(
            "Rolling window length (in drops) to average reward/throughput curves "
            "before plotting."
        ),
    )

    args = parser.parse_args()

    reward_files = _find_reward_files(args.root, args.drops, args.rx_ue, args.tx_ue)

    if not reward_files:
        raise SystemExit(
            f"No reward files found under {args.root} matching the provided filters. "
            "Expected files named deqn_rewards_drop_<id>[_rx_UE_<rx>_tx_UE_<tx>].npz."
        )
    
    imitation_label = _load_imitation_label([info.path for info in reward_files])

    pair_rewards, drop_ids = _aggregate_rewards(reward_files)
    pair_rewards, drop_ids = _apply_rolling_mean(pair_rewards, args.rolling_window)
    plot_rewards(
        pair_rewards, drop_ids, args.output, show=args.show, imitation_label=imitation_label
    )

    action_files = _find_action_files(args.root, args.drops, args.rx_ue, args.tx_ue)
    if action_files:
        pair_actions, max_step = _aggregate_actions(action_files)
        plot_actions(
            pair_actions,
            max_step,
            args.actions_output,
            show=args.show,
            imitation_label=imitation_label,
        )
    else:
        print("No action files found; skipping action plot.")

    throughput_files = _find_throughput_files(args.root, args.drops, args.rx_ue, args.tx_ue)
    if throughput_files:
        pair_throughput, tp_drop_ids = _aggregate_throughput(throughput_files)
        pair_throughput, tp_drop_ids = _apply_rolling_mean(
            pair_throughput, args.rolling_window
        )
        if tp_drop_ids and set(tp_drop_ids) != set(drop_ids):
            print(
                "Warning: Drop ids for throughput and reward plots differ; "
                "plots will include all available data."
            )
        if imitation_label is None:
            imitation_label = _load_imitation_label([info.path for info in throughput_files])
        plot_throughput(
            pair_throughput,
            tp_drop_ids or drop_ids,
            args.throughput_output,
            show=args.show,
            imitation_label=imitation_label,
        )
    else:
        print("No throughput files found; skipping throughput plot.")

if __name__ == "__main__":
    main()