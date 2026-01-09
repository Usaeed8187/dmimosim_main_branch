"""Plot DEQN reward and throughput trends across drops and agent pairs.

This utility scans the results directory for files named
``deqn_rewards_drop_<drop>[_rx_UE_<rx>_tx_UE_<tx>].npz`` and
``deqn_actions_drop_<drop>[_rx_UE_<rx>_tx_UE_<tx>].npz`` (produced by
``sims/sim_mu_mimo_deqn_chan_pred_training.py``) and generates line plots
showing the average reward per (rx, tx) agent pair for each drop, as well as
the action indices chosen at every time step.

Throughput is loaded from files named
``mu_mimo_results_link_adapt_rx_UE_<rx>_tx_UE_<tx>_prediction_deqn_pmi_quantization_True[_imitation_<method>_steps_<steps>].npz``
that live directly inside per-drop folders explicitly supplied on the
command line.

"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


REWARD_PATTERN = re.compile(
    r"deqn_rewards_drop_(\d+)"
    r"(?:_rx_UE_(\d+)_tx_UE_(\d+))?"
    r"_imitation_([A-Za-z0-9_]+)_steps_(\d+)\.npz$"
)

ACTION_PATTERN = re.compile(
    r"deqn_actions_drop_(\d+)"
    r"(?:_rx_UE_(\d+)_tx_UE_(\d+))?"
    r"_imitation_([A-Za-z0-9_]+)_steps_(\d+)\.npz$"
)


THROUGHPUT_PATTERN = re.compile(
    r"mu_mimo_results_link_adapt_rx_UE_(\d+)_tx_UE_(\d+)"
    r"_prediction_deqn_plus_two_mode_pmi_quantization_True"
    r"_imitation_([A-Za-z0-9_]+)_steps_(\d+)\.npz$"
)


@dataclass(frozen=True)
class RewardFile:
    path: Path
    drop_id: int
    rx_ue: Optional[int]
    tx_ue: Optional[int]
    imitation_method: Optional[str]
    imitation_steps: Optional[int]
    

@dataclass(frozen=True)
class ActionFile:
    path: Path
    drop_id: int
    rx_ue: Optional[int]
    tx_ue: Optional[int]
    imitation_method: Optional[str]
    imitation_steps: Optional[int]

@dataclass(frozen=True)
class ThroughputFile:
    path: Path
    drop_id: int
    rx_ue: int
    tx_ue: int
    imitation_method: Optional[str] | None = None
    imitation_steps: Optional[int] | None = None

@dataclass
class RunData:
    label: str
    reward_files: List[RewardFile]
    action_files: List[ActionFile]
    throughput_files: List[ThroughputFile]

def _extract_metadata(path: Path) -> RewardFile:
    """Extract drop/rx/tx identifiers from a reward file name."""


    match = REWARD_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse metadata from {path}")
    drop_id = int(match.group(1))
    rx_ue = int(match.group(2)) if match.group(2) else None
    tx_ue = int(match.group(3)) if match.group(3) else None
    imitation_method = match.group(4) if match.group(4) else None
    imitation_steps = int(match.group(5)) if match.group(5) else None
    return RewardFile(
        path=path,
        drop_id=drop_id,
        rx_ue=rx_ue,
        tx_ue=tx_ue,
        imitation_method=imitation_method,
        imitation_steps=imitation_steps,
    )

def _extract_action_metadata(path: Path) -> ActionFile:
    """Extract drop/rx/tx identifiers from an action log file name."""

    match = ACTION_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse metadata from {path}")
    drop_id = int(match.group(1))
    rx_ue = int(match.group(2)) if match.group(2) else None
    tx_ue = int(match.group(3)) if match.group(3) else None
    imitation_method = match.group(4) if match.group(4) else None
    imitation_steps = int(match.group(5)) if match.group(5) else None
    return ActionFile(
        path=path,
        drop_id=drop_id,
        rx_ue=rx_ue,
        tx_ue=tx_ue,
        imitation_method=imitation_method,
        imitation_steps=imitation_steps,
    )

def _extract_throughput_metadata(path: Path, drop_id: int) -> ThroughputFile:
    """Extract rx/tx/optional imitation identifiers from a throughput filename."""

    match = THROUGHPUT_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse throughput metadata from {path}")

    rx_ue = int(match.group(1))
    tx_ue = int(match.group(2))
    imitation_method = match.group(3) if match.group(3) else None
    imitation_steps = int(match.group(4)) if match.group(4) else None
    return ThroughputFile(
        path=path,
        drop_id=drop_id,
        rx_ue=rx_ue,
        tx_ue=tx_ue,
        imitation_method=imitation_method,
        imitation_steps=imitation_steps,
    )



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
        for path in sorted(drop_path.glob("deqn_rewards_drop_*_imitation_*_steps_*.npz")):
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
                "Info: Multiple reward files found for drop "
                f"{drop}; using ALL {len(candidates)} matches"
            )

        files.extend(candidates)

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
        for path in sorted(drop_path.glob("deqn_actions_drop_*_imitation_*_steps_*.npz")):
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
                "Info: Multiple action files found for drop "
                f"{drop}; using ALL {len(candidates)} matches"
            )

        files.extend(candidates)

    return files

def _find_throughput_files(
    root: Path, drops: Iterable[int], rx_ue: int, tx_ue: int
) -> List[ThroughputFile]:
    """Return throughput result files under ``root`` for specified drops."""

    files: List[ThroughputFile] = []
    for drop in drops:
        drop_path = root / f"channels_high_mobility_{drop}"
        if not drop_path.exists():
            print(f"Warning: Drop directory not found: {drop_path}")
            continue

        candidates: List[ThroughputFile] = []
        for path in sorted(
            drop_path.glob(
                "mu_mimo_results_link_adapt_rx_UE_*_tx_UE_*_prediction_deqn_plus_two_mode_pmi_quantization_True_imitation_*_steps_*.npz"
            )
        ):
            try:
                info = _extract_throughput_metadata(path, drop_id=int(drop))
            except ValueError:
                continue

            if info.rx_ue != rx_ue or info.tx_ue != tx_ue:
                continue
            candidates.append(info)


        if not candidates:
            print(
                "Warning: Throughput file not found for drop "
                f"{drop}: expected rx {rx_ue}, tx {tx_ue} under {drop_path}"
            )
            continue
        if len(candidates) > 1:
            print(
                "Info: Multiple throughput files found for drop "
                f"{drop}; using ALL {len(candidates)} matches"
            )
        files.extend(candidates)

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

def _imitation_label_from_files(files: Iterable[RewardFile | ActionFile]) -> Optional[str]:
    """Generate an imitation label from filename metadata if present."""

    for file_info in files:
        if getattr(file_info, "imitation_method", None) is not None:
            steps = getattr(file_info, "imitation_steps", None)
            method = getattr(file_info, "imitation_method")
            if steps is None:
                return f"Imitation: {method}"
            return f"Imitation: {method} ({steps} steps)"
    return None


def _slugify_label(label: str) -> str:
    """Create a filename-safe suffix from the provided label."""

    safe = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return safe or "run"


def _group_by_imitation(files):
    """Group RewardFile/ActionFile/ThroughputFile by (method, steps)."""
    groups = defaultdict(list)
    for f in files:
        key = (getattr(f, "imitation_method", None), getattr(f, "imitation_steps", None))
        groups[key].append(f)
    return groups


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

def _compute_average_reward(
    pair_rewards: Dict[Tuple[int, int], List[Tuple[int, float]]]
) -> List[Tuple[int, float]]:
    """Compute average reward across all agents for each drop."""

    drop_to_values: DefaultDict[int, List[float]] = defaultdict(list)
    for series in pair_rewards.values():
        for drop, value in series:
            drop_to_values[drop].append(value)

    averaged: List[Tuple[int, float]] = []
    for drop, values in drop_to_values.items():
        averaged.append((drop, float(np.mean(values))))

    averaged.sort(key=lambda item: item[0])
    return averaged

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
    series_by_run: Sequence[Tuple[str, Dict[Tuple[int, int], List[Tuple[int, float]]]]],
    drop_ids: List[int],
    output: Path,
    show: bool = False,
) -> None:
    """Plot mean rewards per (rx, tx) pair across drops for all runs."""

    if not series_by_run:
        raise RuntimeError("No reward data found to plot.")

    plt.figure(figsize=(10, 6))
    for run_label, pair_rewards in series_by_run:
        for (rx_idx, tx_idx), series in sorted(pair_rewards.items()):
            drops, values = zip(*series)
            plt.plot(
                drops,
                values,
                marker="o",
                label=f"{run_label} rx{rx_idx}-tx{tx_idx}",
            )

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

def plot_average_rewards(
    series_by_run: Sequence[Tuple[str, List[Tuple[int, float]]]],
    drop_ids: List[int],
    output: Path,
    show: bool = False,
) -> None:
    """Plot average reward across all agents for each run."""

    if not series_by_run:
        raise RuntimeError("No average reward data found to plot.")

    plt.figure(figsize=(10, 6))
    for run_label, series in series_by_run:
        drops, values = zip(*series)
        plt.plot(drops, values, marker="o", label=run_label)

    plt.xlabel("Drop")
    plt.ylabel("Average reward across agents")
    plt.title("DEQN average rewards across drops")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(drop_ids)
    plt.legend()
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved average reward plot to {output}")

def plot_throughput(
    series_by_run: Sequence[Tuple[str, Dict[Tuple[int, int], List[Tuple[int, float]]]]],
    drop_ids: List[int],
    output: Path,
    show: bool = False,
) -> None:
    """Plot mean throughput per (rx, tx) pair across drops for all runs."""

    if not series_by_run:
        raise RuntimeError("No throughput data found to plot.")

    plt.figure(figsize=(10, 6))
    for run_label, pair_throughput in series_by_run:
        for (rx_idx, tx_idx), series in sorted(pair_throughput.items()):
            drops, values = zip(*series)
            plt.plot(
                drops,
                values,
                marker="s",
                label=f"{run_label} rx{rx_idx}-tx{tx_idx}",
            )

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
        nargs="+",
        default=[Path("results") / "channels_multiple_mu_mimo"],
        help="Root directory/directories to scan for deqn_rewards_drop_*.npz files.",
    )
    parser.add_argument(
        "--drops",
        type=int,
        nargs="+",
        # default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        default=list(range(4, 45)),
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
        "--average-reward-output",
        type=Path,
        default=Path("results") / "deqn_average_rewards.png",
        help="Path to save the generated average reward plot image.",
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
        default=10,
        help=(
            "Rolling window length (in drops) to average reward/throughput curves "
            "before plotting."
        ),
    )

    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help=(
            "Optional labels to use for each root directory; must align with order of --root."
        ),
    )

    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.root):
        raise SystemExit("--labels must match the number of provided --root entries.")

    runs: List[RunData] = []
    for idx, root in enumerate(args.root):
        reward_files_all = _find_reward_files(root, args.drops, args.rx_ue, args.tx_ue)

        if not reward_files_all:
            print(
                "Warning: No reward files found under "
                f"{root} matching the provided filters."
            )
            continue

        action_files_all = _find_action_files(root, args.drops, args.rx_ue, args.tx_ue)
        throughput_files_all = _find_throughput_files(root, args.drops, args.rx_ue, args.tx_ue)

        # Group by imitation setting so we plot multiple curves on the same plots
        rewards_by_key = _group_by_imitation(reward_files_all)
        actions_by_key = _group_by_imitation(action_files_all)
        tp_by_key = _group_by_imitation(throughput_files_all)

        # Base label per root (optional)
        base_label: Optional[str] = None
        if args.labels and idx < len(args.labels):
            base_label = args.labels[idx]
        if base_label is None:
            base_label = root.name

        # Create one RunData per imitation setting
        for (method, steps), reward_files in sorted(rewards_by_key.items()):
            action_files = actions_by_key.get((method, steps), [])
            throughput_files = tp_by_key.get((method, steps), [])

            # Nice label like: "<root> | imitation_two_mode_steps_10"
            if method is None or steps is None:
                run_label = base_label
            else:
                run_label = f"{base_label} | imitation_{method}_steps_{steps}"

            runs.append(
                RunData(
                    label=run_label,
                    reward_files=reward_files,
                    action_files=action_files,
                    throughput_files=throughput_files,
                )
            )

    if not runs:
        raise SystemExit(
            "No reward files found in the provided roots. "
            "Expected files named deqn_rewards_drop_<id>[_rx_UE_<rx>_tx_UE_<tx>].npz."
        )
    
    reward_series: List[Tuple[str, Dict[Tuple[int, int], List[Tuple[int, float]]]]] = []
    avg_reward_series: List[Tuple[str, List[Tuple[int, float]]]] = []
    drop_union: set[int] = set()
    
    for run in runs:
        pair_rewards, drop_ids = _aggregate_rewards(run.reward_files)
        pair_rewards, drop_ids = _apply_rolling_mean(pair_rewards, args.rolling_window)
        if not pair_rewards:
            print(f"Warning: No reward data found for run {run.label}")
            continue
        reward_series.append((run.label, pair_rewards))
        avg_reward_series.append((run.label, _compute_average_reward(pair_rewards)))
        drop_union.update(drop_ids)

    if reward_series:
        drop_ids_sorted = sorted(drop_union)
        plot_rewards(reward_series, drop_ids_sorted, args.output, show=args.show)
        plot_average_rewards(
            avg_reward_series, drop_ids_sorted, args.average_reward_output, show=args.show
        )
    else:
        print("No reward data aggregated; skipping reward plots.")

    for run in runs:
        if not run.action_files:
            continue
        pair_actions, max_step = _aggregate_actions(run.action_files)
        if not pair_actions:
            continue
        action_output = args.actions_output
        if len(runs) > 1:
            stem = args.actions_output.stem
            action_output = args.actions_output.with_name(
                f"{stem}_{_slugify_label(run.label)}{args.actions_output.suffix}"
            )
        plot_actions(
            pair_actions,
            max_step,
            action_output,
            show=args.show,
            imitation_label=run.label,
        )
    throughput_series: List[Tuple[str, Dict[Tuple[int, int], List[Tuple[int, float]]]]] = []
    tp_drop_union: set[int] = set()
    for run in runs:
        if not run.throughput_files:
            continue
        pair_throughput, tp_drop_ids = _aggregate_throughput(run.throughput_files)
        pair_throughput, tp_drop_ids = _apply_rolling_mean(
            pair_throughput, args.rolling_window
        )
        throughput_series.append((run.label, pair_throughput))
        tp_drop_union.update(tp_drop_ids)

    if throughput_series:
        drop_ids_sorted = sorted(tp_drop_union or drop_union)
        plot_throughput(throughput_series, drop_ids_sorted, args.throughput_output, show=args.show)
    else:
        print("No throughput files found; skipping throughput plot.")

if __name__ == "__main__":
    main()