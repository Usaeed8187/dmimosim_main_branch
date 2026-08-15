#!/usr/bin/env python3
"""Aggregate and plot predictor complexity instrumentation from result NPZs.

The script consumes ``predictor_complexity_raw_json`` written by
``sims/sim_mu_mimo_testing_updates.py``.  Raw prediction-event samples are
pooled across drops before percentiles are evaluated.  Configuration latency
is amortized over the actual number of online prediction events unless an
explicit horizon is supplied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "steady_state_kalman_filter": "Steady-State KF",
    "wesn_lite": "Low-Rank Configured WESN",
    "configured_wesn": "Configured WESN",
}
METHOD_STYLES = {
    "steady_state_kalman_filter": {"color": "tab:purple", "marker": "P"},
    "wesn_lite": {"color": "tab:brown", "marker": "v"},
    "configured_wesn": {"color": "tab:green", "marker": "^"},
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
    }
)


@dataclass(frozen=True)
class Artifact:
    path: Path
    mobility: str
    drop: int
    rx_ues: int
    tx_ues: int
    method: str
    workers: int
    split_mode: str
    raw: dict

    @property
    def num_rus(self) -> int:
        # num_txue_sel excludes the central/gNB RU in the existing figures.
        return self.tx_ues + 1

    def phase_records(self, phase: str) -> list[dict]:
        return list(self.raw.get("phases", {}).get(phase, []))


def _json_scalar(npz, key: str) -> dict | None:
    if key not in npz.files:
        return None
    try:
        return json.loads(str(npz[key].item()))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _split_mode(path: Path) -> str:
    name = path.stem
    if "_drop_split" in name:
        return "drop_split"
    if "_time_split" in name:
        return "time_split"
    return "unspecified"


def load_artifact(path: Path) -> Artifact | None:
    folder_match = re.fullmatch(r"channels_(.+)_(\d+)", path.parent.name)
    method_match = re.search(r"_prediction_(.+?)_pmi_quantization_", path.name)
    rx_match = re.search(r"_rx_UE_(\d+)_", path.name)
    tx_match = re.search(r"_tx_UE_(\d+)_", path.name)
    if not (folder_match and method_match and rx_match and tx_match):
        return None

    try:
        with np.load(path, allow_pickle=False) as npz:
            raw = _json_scalar(npz, "predictor_complexity_raw_json")
            if raw is None:
                return None
            workers = int(npz["predictor_workers"].item()) if "predictor_workers" in npz.files else 1
    except (OSError, ValueError, KeyError):
        return None

    return Artifact(
        path=path,
        mobility=folder_match.group(1),
        drop=int(folder_match.group(2)),
        rx_ues=int(rx_match.group(1)),
        tx_ues=int(tx_match.group(1)),
        method=method_match.group(1),
        workers=workers,
        split_mode=_split_mode(path),
        raw=raw,
    )


def discover_artifacts(
    base_dir: Path,
    methods: set[str],
    mobilities: set[str] | None,
    rx_ues: int | None,
    split_mode: str,
    include_fixed_mcs: bool,
) -> tuple[list[Artifact], int]:
    artifacts: list[Artifact] = []
    skipped_uninstrumented = 0
    for path in sorted(base_dir.glob("channels_*/*.npz")):
        if "_prediction_" not in path.name:
            continue
        if not include_fixed_mcs and "_link_adapt_" not in path.name:
            continue
        method_match = re.search(r"_prediction_(.+?)_pmi_quantization_", path.name)
        if method_match is None or method_match.group(1) not in methods:
            continue
        artifact = load_artifact(path)
        if artifact is None:
            skipped_uninstrumented += 1
            continue
        if mobilities is not None and artifact.mobility not in mobilities:
            continue
        if rx_ues is not None and artifact.rx_ues != rx_ues:
            continue
        if split_mode != "all" and artifact.split_mode not in (split_mode, "unspecified"):
            continue
        artifacts.append(artifact)
    return artifacts, skipped_uninstrumented


def pooled_latency(artifacts: Sequence[Artifact]) -> np.ndarray:
    values = [
        float(record["elapsed_seconds"])
        for artifact in artifacts
        for record in artifact.phase_records("inference_system")
    ]
    return np.asarray(values, dtype=float)


def configuration_latencies(artifacts: Sequence[Artifact]) -> np.ndarray:
    values = [
        float(record["elapsed_seconds"])
        for artifact in artifacts
        for record in artifact.phase_records("configuration_system")
    ]
    return np.asarray(values, dtype=float)


def max_phase_value(artifact: Artifact, field: str) -> float:
    records = [record for rows in artifact.raw.get("phases", {}).values() for record in rows]
    return max((float(record.get(field, 0.0)) for record in records), default=0.0)


def group_artifacts(artifacts: Iterable[Artifact]):
    grouped = defaultdict(list)
    for artifact in artifacts:
        grouped[(artifact.mobility, artifact.method, artifact.num_rus, artifact.workers)].append(artifact)
    return grouped


def summarize_groups(grouped: dict, amortization_horizon: int | None) -> list[dict]:
    rows: list[dict] = []
    for (mobility, method, num_rus, workers), artifacts in sorted(grouped.items()):
        latency = pooled_latency(artifacts)
        configuration = configuration_latencies(artifacts)
        if latency.size == 0:
            continue

        persistent = np.asarray(
            [float(a.raw.get("persistent_predictor_bytes", 0.0)) for a in artifacts],
            dtype=float,
        )
        workspace = np.asarray(
            [max_phase_value(a, "python_peak_increment_bytes") for a in artifacts],
            dtype=float,
        )
        peak_rss = np.asarray(
            [max_phase_value(a, "process_peak_rss_bytes") for a in artifacts],
            dtype=float,
        )
        residue_rank_histogram: dict[int, int] = {}
        residue_energy_thresholds = []
        for artifact in artifacts:
            rank_summary = artifact.raw.get("wesn_residue_rank_summary", {})
            for rank, count in rank_summary.get("histogram", {}).items():
                rank_int = int(rank)
                residue_rank_histogram[rank_int] = (
                    residue_rank_histogram.get(rank_int, 0) + int(count)
                )
            if "energy_threshold" in rank_summary:
                residue_energy_thresholds.append(
                    float(rank_summary["energy_threshold"])
                )
        rank_count = sum(residue_rank_histogram.values())
        residue_rank_mean = (
            sum(rank * count for rank, count in residue_rank_histogram.items())
            / rank_count
            if rank_count
            else np.nan
        )
        residue_rank_mode = (
            min(
                residue_rank_histogram,
                key=lambda rank: (-residue_rank_histogram[rank], rank),
            )
            if rank_count
            else np.nan
        )

        inference_mean = float(np.mean(latency))
        if amortization_horizon is None:
            amortized = float((np.sum(latency) + np.sum(configuration)) / latency.size)
            horizon_label = "actual_online_events"
        else:
            amortized = inference_mean + float(np.mean(configuration)) / amortization_horizon
            horizon_label = str(amortization_horizon)

        rows.append(
            {
                "mobility": mobility,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "num_rus": num_rus,
                "workers": workers,
                "num_drops": len({a.drop for a in artifacts}),
                "num_latency_samples": int(latency.size),
                "latency_mean_ms": 1e3 * inference_mean,
                "latency_p50_ms": 1e3 * float(np.percentile(latency, 50)),
                "latency_p95_ms": 1e3 * float(np.percentile(latency, 95)),
                "latency_p99_ms": 1e3 * float(np.percentile(latency, 99)),
                "configuration_mean_ms": 1e3 * float(np.mean(configuration)) if configuration.size else 0.0,
                "amortization_horizon": horizon_label,
                "amortized_latency_ms": 1e3 * amortized,
                "persistent_memory_mean_mib": float(np.mean(persistent)) / 2**20,
                "persistent_memory_max_mib": float(np.max(persistent)) / 2**20,
                "workspace_peak_mib": float(np.max(workspace)) / 2**20,
                "process_peak_rss_mib": float(np.max(peak_rss)) / 2**20,
                "residue_rank_mean": residue_rank_mean,
                "residue_rank_mode": residue_rank_mode,
                "residue_rank_count": rank_count,
                "residue_energy_threshold": (
                    float(np.mean(residue_energy_thresholds))
                    if residue_energy_thresholds
                    else np.nan
                ),
            }
        )
    return rows


def parallel_rows(summary_rows: Sequence[dict]) -> list[dict]:
    by_scenario = defaultdict(dict)
    for row in summary_rows:
        by_scenario[(row["mobility"], row["method"], row["num_rus"])][row["workers"]] = row

    rows = []
    for (mobility, method, num_rus), worker_rows in sorted(by_scenario.items()):
        baseline = worker_rows.get(1)
        if baseline is None:
            continue
        baseline_latency = baseline["latency_p50_ms"]
        for workers, row in sorted(worker_rows.items()):
            latency = row["latency_p50_ms"]
            speedup = baseline_latency / latency if latency > 0 else np.nan
            rows.append(
                {
                    "mobility": mobility,
                    "method": method,
                    "method_label": row["method_label"],
                    "num_rus": num_rus,
                    "workers": workers,
                    "latency_p50_ms": latency,
                    "speedup": speedup,
                    "efficiency": speedup / workers,
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _method_rows(rows: Sequence[dict], mobility: str, workers: int, method: str):
    return sorted(
        [r for r in rows if r["mobility"] == mobility and r["workers"] == workers and r["method"] == method],
        key=lambda r: r["num_rus"],
    )


def plot_latency(rows: Sequence[dict], mobility: str, workers: int, methods: Sequence[str], output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), sharex=True)
    for method in methods:
        selected = _method_rows(rows, mobility, workers, method)
        if not selected:
            continue
        style = METHOD_STYLES.get(method, {"marker": "o"})
        x = [r["num_rus"] for r in selected]
        for axis, field, title in zip(
            axes,
            ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms"),
            ("Median", "95th percentile", "99th percentile"),
        ):
            axis.plot(x, [r[field] for r in selected], label=METHOD_LABELS.get(method, method), **style)
            axis.set_title(title)
            axis.set_xlabel("Number of RUs")
    axes[0].set_ylabel("System predictor latency (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(handles)), bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"{mobility.replace('_', ' ').title()}, Q={workers}", y=1.16)
    save_figure(fig, output_dir / f"latency_quantiles_{mobility}_workers_{workers}")


def plot_memory(rows: Sequence[dict], mobility: str, workers: int, methods: Sequence[str], output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), sharex=True)
    fields = (
        ("persistent_memory_mean_mib", "Persistent predictor memory (MiB)", "Persistent"),
        ("workspace_peak_mib", "Peak measured workspace (MiB)", "Workspace"),
        ("process_peak_rss_mib", "Peak process RSS (MiB)", "Process RSS"),
    )
    for method in methods:
        selected = _method_rows(rows, mobility, workers, method)
        if not selected:
            continue
        style = METHOD_STYLES.get(method, {"marker": "o"})
        x = [r["num_rus"] for r in selected]
        for axis, (field, ylabel, title) in zip(axes, fields):
            axis.plot(x, [r[field] for r in selected], label=METHOD_LABELS.get(method, method), **style)
            axis.set_title(title)
            axis.set_xlabel("Number of RUs")
            axis.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(handles)), bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"{mobility.replace('_', ' ').title()}, Q={workers}", y=1.16)
    save_figure(fig, output_dir / f"memory_{mobility}_workers_{workers}")


def plot_amortized(rows: Sequence[dict], mobility: str, workers: int, methods: Sequence[str], output_dir: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for method in methods:
        selected = _method_rows(rows, mobility, workers, method)
        if not selected:
            continue
        style = METHOD_STYLES.get(method, {"marker": "o"})
        x = [r["num_rus"] for r in selected]
        ax.plot(x, [r["amortized_latency_ms"] for r in selected], label=f"{METHOD_LABELS.get(method, method)}: amortized", **style)
        ax.plot(
            x,
            [r["latency_mean_ms"] for r in selected],
            label=f"{METHOD_LABELS.get(method, method)}: inference/update",
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle="--",
            alpha=0.7,
        )
    ax.set_xlabel("Number of RUs")
    ax.set_ylabel("Latency per online prediction event (ms)")
    ax.set_title(f"Amortized latency: {mobility.replace('_', ' ').title()}, Q={workers}")
    ax.legend()
    save_figure(fig, output_dir / f"amortized_latency_{mobility}_workers_{workers}")


def plot_parallel(rows: Sequence[dict], mobility: str, num_rus: int, methods: Sequence[str], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True)
    found = False
    for method in methods:
        selected = sorted(
            [r for r in rows if r["mobility"] == mobility and r["num_rus"] == num_rus and r["method"] == method],
            key=lambda r: r["workers"],
        )
        if not selected:
            continue
        found = True
        style = METHOD_STYLES.get(method, {"marker": "o"})
        workers = [r["workers"] for r in selected]
        axes[0].plot(workers, [r["speedup"] for r in selected], label=METHOD_LABELS.get(method, method), **style)
        axes[1].plot(workers, [r["efficiency"] for r in selected], label=METHOD_LABELS.get(method, method), **style)
    if not found:
        plt.close(fig)
        return
    all_workers = sorted({r["workers"] for r in rows if r["mobility"] == mobility and r["num_rus"] == num_rus})
    if all_workers:
        axes[0].plot(all_workers, all_workers, color="0.5", linestyle=":", label="Ideal")
    axes[0].set_ylabel("Speedup $S(Q)$")
    axes[1].set_ylabel("Parallel efficiency $E(Q)$")
    for axis in axes:
        axis.set_xlabel("Predictor workers $Q$")
        axis.set_xticks(all_workers)
    axes[1].set_ylim(bottom=0.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(handles)), bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"{mobility.replace('_', ' ').title()}, {num_rus} RUs", y=1.16)
    save_figure(fig, output_dir / f"parallel_efficiency_{mobility}_rus_{num_rus}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "channels_multiple_mu_mimo",
        help="Directory containing channels_<mobility>_<drop> result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "predictor_complexity",
        help="Destination for CSV, PDF, and PNG outputs.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["steady_state_kalman_filter", "wesn_lite"],
    )
    parser.add_argument("--mobilities", nargs="+", default=None)
    parser.add_argument("--rx-ues", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker count used for latency, memory, and amortized plots.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("time_split", "drop_split", "all"),
        default="time_split",
        help="Unspecified baseline files are retained for either explicit split.",
    )
    parser.add_argument(
        "--amortization-horizon",
        type=int,
        default=None,
        help="Override H; default amortizes each configuration over its actual saved online events.",
    )
    parser.add_argument(
        "--include-fixed-mcs",
        action="store_true",
        help="Include fixed-MCS artifacts in addition to the default link-adaptation sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.amortization_horizon is not None and args.amortization_horizon <= 0:
        raise ValueError("--amortization-horizon must be positive.")

    artifacts, skipped = discover_artifacts(
        base_dir=args.base_dir,
        methods=set(args.methods),
        mobilities=set(args.mobilities) if args.mobilities else None,
        rx_ues=args.rx_ues,
        split_mode=args.split_mode,
        include_fixed_mcs=args.include_fixed_mcs,
    )
    if not artifacts:
        raise SystemExit(
            "No instrumented result archives matched. Run the SS-KF/low-rank WESN sweep first "
            "and check --base-dir, --mobilities, --rx-ues, and --split-mode."
        )

    grouped = group_artifacts(artifacts)
    summary = summarize_groups(grouped, args.amortization_horizon)
    parallel = parallel_rows(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "complexity_summary.csv", summary)
    write_csv(args.output_dir / "parallel_efficiency.csv", parallel)

    mobilities = sorted({artifact.mobility for artifact in artifacts})
    for mobility in mobilities:
        plot_latency(summary, mobility, args.workers, args.methods, args.output_dir)
        plot_memory(summary, mobility, args.workers, args.methods, args.output_dir)
        plot_amortized(summary, mobility, args.workers, args.methods, args.output_dir)
        for num_rus in sorted({artifact.num_rus for artifact in artifacts if artifact.mobility == mobility}):
            plot_parallel(parallel, mobility, num_rus, args.methods, args.output_dir)

    print(f"Loaded {len(artifacts)} instrumented archives; skipped {skipped} uninstrumented/unreadable archives.")
    print(f"Wrote {len(summary)} aggregate rows to {args.output_dir / 'complexity_summary.csv'}")
    print(f"Wrote {len(parallel)} parallel rows to {args.output_dir / 'parallel_efficiency.csv'}")
    print(f"Plots written under {args.output_dir}")


if __name__ == "__main__":
    main()
