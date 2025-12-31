"""Plot DEQN reward trends across drops and agent pairs.

This utility scans the results directory for files named
``deqn_rewards_drop_<drop>.npz`` (produced by
``sims/sim_mu_mimo_deqn_chan_pred_training.py``) and generates a line plot
showing the average reward per (rx, tx) agent pair for each drop.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REWARD_PATTERN = re.compile(r"deqn_rewards_drop_(\d+)\.npz$")


def _find_reward_files(root: Path) -> List[Path]:
    """Return all DEQN reward files under ``root``."""

    return sorted(root.rglob("deqn_rewards_drop_*.npz"))


def _extract_drop_id(path: Path) -> int:
    """Extract the numeric drop identifier from a reward file name."""

    match = REWARD_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse drop id from {path}")
    return int(match.group(1))


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


def _aggregate_rewards(files: Iterable[Path]) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, float]]], List[int]]:
    """Compute per-drop average rewards for each (rx, tx) pair.

    Returns:
        pair_rewards: mapping from (rx_idx, tx_idx) to a list of (drop, mean_reward)
        drop_ids: sorted list of all drop identifiers encountered
    """

    pair_rewards: DefaultDict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
    drop_ids: List[int] = []

    for file_path in files:
        drop_id = _extract_drop_id(file_path)
        rewards = _load_rewards(file_path)

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
        "--show",
        action="store_true",
        help="Display the plot window after saving.",
    )
    args = parser.parse_args()

    reward_files = _find_reward_files(args.root)
    if not reward_files:
        raise SystemExit(
            f"No reward files found under {args.root}. Expected files named deqn_rewards_drop_<id>.npz."
        )

    pair_rewards, drop_ids = _aggregate_rewards(reward_files)
    plot_rewards(pair_rewards, drop_ids, args.output, show=args.show)


if __name__ == "__main__":
    main()