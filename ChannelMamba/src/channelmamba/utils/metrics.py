"""Metrics and result serialization helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import torch


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def maybe_compute_flops(model: torch.nn.Module, dummy_input: torch.Tensor) -> str | None:
    try:
        from thop import clever_format, profile
    except ImportError:
        return None
    try:
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops_formatted, _ = clever_format([flops, 0], "%.3f")
        return flops_formatted
    except Exception:
        return None


def write_json(payload: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return target


def write_results_csv(rows: Iterable[dict], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return target
    fieldnames = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target
