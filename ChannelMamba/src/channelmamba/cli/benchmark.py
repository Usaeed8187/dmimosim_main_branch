"""Run all released evaluation suites for a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..config import load_experiment_config
from ..evaluation import suite_to_rows
from ..models import ChannelMamba
from ..utils.metrics import count_parameters, maybe_compute_flops, write_json, write_results_csv
from ..utils.runtime import dump_config_snapshot, ensure_dir, prepare_output_dir, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ChannelMamba across all public suites.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    parser.add_argument("--duplex", required=True, choices=["tdd", "fdd"], help="Benchmark duplex mode.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--device", default=None, help="Runtime device, e.g. cpu/cuda/auto.")
    parser.add_argument("--output-dir", default=None, help="Override output root directory.")
    parser.add_argument("--run-name", default=None, help="Override run name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.device is not None:
        config.runtime.device = args.device
    if args.output_dir is not None:
        config.runtime.output_root = args.output_dir
    if args.run_name is not None:
        config.runtime.run_name = args.run_name

    device = select_device(config.runtime.device)
    output_dir = prepare_output_dir(config.runtime.output_root, config.runtime.run_name or f"benchmark_{args.duplex}")
    metrics_dir = ensure_dir(output_dir / "metrics")
    dump_config_snapshot(config.to_dict(), output_dir)

    model = ChannelMamba.from_config(config.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    trainable_params, _ = count_parameters(model)
    dummy_input = torch.randn(1, config.model.prev_len, config.model.d_in, device=device)
    flops = maybe_compute_flops(model, dummy_input)

    all_rows: list[dict] = []
    for suite_name in ("single", "snr", "velocity"):
        rows = suite_to_rows(
            suite_name=suite_name,
            model=model,
            data_root=args.data_root,
            duplex=args.duplex,
            data_config=config.data,
            eval_config=config.eval,
            device=device,
        )
        for row in rows:
            row["params_m"] = round(trainable_params / 1_000_000, 4)
            row["flops"] = flops or "N/A"
        write_results_csv(rows, metrics_dir / f"{suite_name}.csv")
        all_rows.extend(rows)

    write_json(
        {
            "duplex": args.duplex,
            "device": str(device),
            "checkpoint": args.checkpoint,
            "rows": all_rows,
            "flops": flops,
        },
        metrics_dir / "benchmark.json",
    )
    print(f"Benchmark artifacts saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
