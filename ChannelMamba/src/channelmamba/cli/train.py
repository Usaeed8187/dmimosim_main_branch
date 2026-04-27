"""Unified training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..config import load_experiment_config
from ..data import build_train_val_datasets
from ..losses import NMSELoss
from ..models import ChannelMamba
from ..utils.metrics import count_parameters, maybe_compute_flops, write_json, write_results_csv
from ..utils.runtime import dump_config_snapshot, ensure_dir, prepare_output_dir, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ChannelMamba.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    parser.add_argument("--duplex", required=True, choices=["tdd", "fdd"], help="Training duplex mode.")
    parser.add_argument("--device", default=None, help="Runtime device, e.g. cpu/cuda/auto.")
    parser.add_argument("--output-dir", default=None, help="Override output root directory.")
    parser.add_argument("--run-name", default=None, help="Override run name.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional smoke-test cap.")
    return parser.parse_args()


def _resolve_train_target(duplex: str, data_config) -> tuple[str, str]:
    if duplex == "tdd":
        return data_config.train_tdd_target_path, data_config.train_tdd_target_key
    return data_config.train_fdd_target_path, data_config.train_fdd_target_key


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer, scheduler, epoch: int, best_val_loss: float, config: dict) -> None:
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": actual_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.device is not None:
        config.runtime.device = args.device
    if args.output_dir is not None:
        config.runtime.output_root = args.output_dir
    if args.run_name is not None:
        config.runtime.run_name = args.run_name
    if args.max_train_batches is not None:
        config.train.max_train_batches = args.max_train_batches
    if args.max_val_batches is not None:
        config.train.max_val_batches = args.max_val_batches

    set_seed(config.train.seed)
    device = select_device(config.runtime.device)
    output_dir = prepare_output_dir(config.runtime.output_root, config.runtime.run_name or f"train_{args.duplex}")
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    dump_config_snapshot(config.to_dict(), output_dir)

    input_path = str(Path(args.data_root) / config.data.train_history_path)
    target_rel_path, target_key = _resolve_train_target(args.duplex, config.data)
    target_path = str(Path(args.data_root) / target_rel_path)

    train_dataset, val_dataset = build_train_val_datasets(
        input_path=input_path,
        target_path=target_path,
        input_key=config.data.train_history_key,
        target_key=target_key,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        group_size=config.data.antenna_group_size,
        noise_min_snr_db=config.data.train_noise_min_snr_db,
        noise_max_snr_db=config.data.train_noise_max_snr_db,
        seed=config.train.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.runtime.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ChannelMamba.from_config(config.model).to(device)
    if device.type == "cuda" and config.runtime.use_data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = NMSELoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=config.train.eta_min,
    )

    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    trainable_params, total_params = count_parameters(actual_model)
    dummy_input = torch.randn(1, config.model.prev_len, config.model.d_in, device=device)
    flops = maybe_compute_flops(actual_model, dummy_input)

    history_rows: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(config.train.epochs):
        model.train()
        train_losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

            if step % config.runtime.log_every == 0:
                print(
                    f"[train] epoch={epoch + 1}/{config.train.epochs} "
                    f"step={step}/{len(train_loader)} loss={loss.item():.6f}"
                )
            if config.train.max_train_batches is not None and step >= config.train.max_train_batches:
                break

        scheduler.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for step, batch in enumerate(val_loader, start=1):
                inputs = batch["inputs"].to(device)
                targets = batch["targets"].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if not torch.isnan(loss):
                    val_losses.append(float(loss.item()))
                if config.train.max_val_batches is not None and step >= config.train.max_val_batches:
                    break

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else float("inf")
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_nmse": avg_train_loss,
                "val_nmse": avg_val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"[epoch] {epoch + 1}/{config.train.epochs} "
            f"train_nmse={avg_train_loss:.6f} val_nmse={avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            _save_checkpoint(
                checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
                config=config.to_dict(),
            )

    if config.runtime.save_last:
        _save_checkpoint(
            checkpoints_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=config.train.epochs,
            best_val_loss=best_val_loss,
            config=config.to_dict(),
        )

    summary = {
        "duplex": args.duplex,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_epoch": best_epoch,
        "best_val_nmse": best_val_loss,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "flops": flops,
        "best_checkpoint": str(checkpoints_dir / "best.pt"),
        "last_checkpoint": str(checkpoints_dir / "last.pt"),
    }
    write_results_csv(history_rows, metrics_dir / "train_history.csv")
    write_json(summary, metrics_dir / "train_summary.json")
    print(f"Training artifacts saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
