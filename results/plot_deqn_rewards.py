"""Plot DEQN reward and throughput trends across drops and agent pairs.

This utility scans the results directory for files named
``deqn_rewards_drop_<drop>[_rx_UE_<rx>_tx_UE_<tx>].npz`` (produced by
``sims/sim_mu_mimo_deqn_chan_pred_training.py``) and generates a line plot
showing the average reward per (rx, tx) agent pair for each drop.
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

THROUGHPUT_PATTERN = re.compile(
    r"mu_mimo_results_.*_rx_UE_(\d+)_tx_UE_(\d+).*.npz$"
)

@dataclass(frozen=True)
class RewardFile:
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

def _infer_drop_id_from_path(path: Path) -> int:
    """Infer a drop identifier from a path component.

    The simulation results are stored under directories whose names contain
    ``drop_<id>``; this helper walks up the directory tree to find that id.
    """

    for part in path.parts:
        match = re.search(r"drop[_-]?(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer drop id from {path}")


def _extract_throughput_metadata(path: Path) -> ThroughputFile:
    """Extract drop/rx/tx identifiers from a throughput result filename."""

    match = THROUGHPUT_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse throughput metadata from {path}")

    rx_ue = int(match.group(1))
    tx_ue = int(match.group(2))
    drop_id = _infer_drop_id_from_path(path)
    return ThroughputFile(path=path, drop_id=drop_id, rx_ue=rx_ue, tx_ue=tx_ue)



def _find_reward_files(root: Path, rx_ue: int | None, tx_ue: int | None) -> List[RewardFile]:
    """Return all DEQN reward files under ``root`` matching the filters."""

    files: List[RewardFile] = []
    for path in sorted(root.rglob("deqn_rewards_drop_*.npz")):
        info = _extract_metadata(path)

        if rx_ue is not None and info.rx_ue != rx_ue:
            continue
        if tx_ue is not None and info.tx_ue != tx_ue:
            continue

        files.append(info)

    return files

def _find_throughput_files(
    root: Path, rx_ue: int | None, tx_ue: int | None
) -> List[ThroughputFile]:
    """Return all throughput result files under ``root`` matching the filters."""

    files: List[ThroughputFile] = []
    for path in sorted(root.rglob("mu_mimo_results_*_rx_UE_*_tx_UE_*.npz")):
        info = _extract_throughput_metadata(path)

        if rx_ue is not None and info.rx_ue != rx_ue:
            continue
        if tx_ue is not None and info.tx_ue != tx_ue:
            continue

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


def plot_rewards(
    pair_rewards: Dict[Tuple[int, int], List[Tuple[int, float]]],
    drop_ids: List[int],
    output: Path,
    show: bool = False,
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
    plt.title("DEQN per-agent rewards across drops")
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
    plt.title("DEQN per-agent throughput across drops")
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
        "--output",
        type=Path,
        default=Path("results") / "deqn_rewards.png",
        help="Path to save the generated plot image.",
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
        default=0,
        help="If provided, only plot rewards saved for this RX UE selection.",
    )
    parser.add_argument(
        "--tx-ue",
        type=int,
        default=0,
        help="If provided, only plot rewards saved for this TX UE selection.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving.",
    )
    args = parser.parse_args()

    reward_files = _find_reward_files(args.root, args.rx_ue, args.tx_ue)
    if not reward_files:
        raise SystemExit(
            f"No reward files found under {args.root} matching the provided filters. "
            "Expected files named deqn_rewards_drop_<id>[_rx_UE_<rx>_tx_UE_<tx>].npz."
        )

    pair_rewards, drop_ids = _aggregate_rewards(reward_files)
    plot_rewards(pair_rewards, drop_ids, args.output, show=args.show)

    throughput_files = _find_throughput_files(args.root, args.rx_ue, args.tx_ue)
    if throughput_files:
        pair_throughput, tp_drop_ids = _aggregate_throughput(throughput_files)
        if tp_drop_ids and set(tp_drop_ids) != set(drop_ids):
            print(
                "Warning: Drop ids for throughput and reward plots differ; "
                "plots will include all available data."
            )
        plot_throughput(
            pair_throughput,
            tp_drop_ids or drop_ids,
            args.throughput_output,
            show=args.show,
        )
    else:
        print("No throughput files found; skipping throughput plot.")

if __name__ == "__main__":
    main()