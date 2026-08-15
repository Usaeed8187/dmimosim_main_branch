"""Lightweight predictor timing and memory instrumentation.

The recorder deliberately keeps channel-history acquisition, configuration,
online adaptation, and inference as separate phases.  RSS values cover native
NumPy allocations; tracemalloc values additionally expose Python workspace.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import resource
import time
import tracemalloc

import numpy as np


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return 0


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return peak * 1024 if os.uname().sysname != "Darwin" else peak


def ensure_metrics(owner) -> dict:
    metrics = getattr(owner, "predictor_complexity_metrics", None)
    if metrics is None:
        metrics = {"schema_version": 1, "phases": {}}
        owner.predictor_complexity_metrics = metrics
    return metrics


@contextmanager
def measure_phase(owner, phase: str, **metadata):
    """Record wall time and process/Python memory for one named phase."""
    metrics = ensure_metrics(owner)
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    tracemalloc.reset_peak()
    python_before, _ = tracemalloc.get_traced_memory()
    rss_before = _rss_bytes()
    start_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed_ns = time.perf_counter_ns() - start_ns
        python_after, python_peak = tracemalloc.get_traced_memory()
        record = {
            "elapsed_seconds": elapsed_ns / 1e9,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": _rss_bytes(),
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "python_current_delta_bytes": python_after - python_before,
            "python_peak_increment_bytes": max(0, python_peak - python_before),
        }
        record.update(metadata)
        metrics["phases"].setdefault(phase, []).append(record)


def phase_summary(metrics: dict) -> dict:
    """Convert raw phase records into serializable latency/memory summaries."""
    summary = {"schema_version": metrics.get("schema_version", 1), "phases": {}}
    for phase, records in metrics.get("phases", {}).items():
        if not records:
            continue
        latency = np.asarray([r["elapsed_seconds"] for r in records], dtype=float)
        summary["phases"][phase] = {
            "samples": int(latency.size),
            "total_seconds": float(np.sum(latency)),
            "mean_seconds": float(np.mean(latency)),
            "p50_seconds": float(np.percentile(latency, 50)),
            "p95_seconds": float(np.percentile(latency, 95)),
            "p99_seconds": float(np.percentile(latency, 99)),
            "max_python_peak_increment_bytes": int(
                max(r["python_peak_increment_bytes"] for r in records)
            ),
            "max_process_peak_rss_bytes": int(
                max(r["process_peak_rss_bytes"] for r in records)
            ),
        }
    for key, value in metrics.items():
        if key not in ("schema_version", "phases"):
            summary[key] = value
    return summary


def numpy_storage_bytes(owner) -> int:
    """Return persistent NumPy array storage directly owned by an object."""
    return int(
        sum(value.nbytes for value in vars(owner).values() if isinstance(value, np.ndarray))
    )
