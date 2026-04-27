"""Evaluation suites for ChannelMamba."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch
from einops import rearrange

from .data import prepare_eval_tensors, transform_tdd_fdd
from .losses import NMSELoss, SpectralEfficiencyLoss


@torch.no_grad()
def evaluate_condition(
    model: torch.nn.Module,
    prev_data: torch.Tensor,
    pred_data: torch.Tensor,
    batch_size: int,
    pred_len: int,
    device: torch.device,
    se_reference_snr_db: float,
) -> dict[str, float]:
    criterion = NMSELoss()
    criterion_se = SpectralEfficiencyLoss(snr_db=se_reference_snr_db, device=device)

    nmse_values: list[float] = []
    se_values: list[float] = []
    se0_values: list[float] = []
    inference_ms: list[float] = []

    total_batches = prev_data.shape[0]
    for start in range(0, total_batches, batch_size):
        end = min(start + batch_size, total_batches)
        current_prev = prev_data[start:end].to(device)
        current_pred = pred_data[start:end].to(device)
        current_bs = current_prev.shape[0]

        current_prev = rearrange(current_prev, "b m l k -> (b m) l k")
        current_pred = rearrange(current_pred, "b m l k -> (b m) l k")

        begin = time.time()
        outputs = model(current_prev)
        inference_ms.append((time.time() - begin) * 1000.0)

        nmse_values.append(float(criterion(outputs, current_pred).item()))
        outputs_re = rearrange(outputs, "(b m) l k -> b l (k m)", b=current_bs)
        target_re = rearrange(current_pred, "(b m) l k -> b l (k m)", b=current_bs)
        se, se0 = criterion_se(
            prediction=transform_tdd_fdd(outputs_re, n_t=4 * 4, n_r=1),
            target=transform_tdd_fdd(target_re, n_t=4 * 4, n_r=1),
        )
        se_values.append(float(se.item()))
        se0_values.append(float(se0.item()))

    se_ratio = float(np.mean(se_values) / np.mean(se0_values)) if se_values and np.mean(se0_values) != 0 else float("nan")
    return {
        "nmse": float(np.mean(nmse_values)) if nmse_values else float("nan"),
        "se_ratio": se_ratio,
        "avg_inference_ms": float(np.mean(inference_ms)) if inference_ms else float("nan"),
    }


def run_single_suite(
    model: torch.nn.Module,
    data_root: str,
    duplex: str,
    data_config,
    eval_config,
    device: torch.device,
) -> list[dict]:
    input_path = f"{data_root}/{data_config.test_history_path}"
    if duplex == "tdd":
        target_path = f"{data_root}/{data_config.test_tdd_target_path}"
        target_key = data_config.test_tdd_target_key
    else:
        target_path = f"{data_root}/{data_config.test_fdd_target_path}"
        target_key = data_config.test_fdd_target_key
    prev_data, pred_data = prepare_eval_tensors(
        input_path=input_path,
        target_path=target_path,
        input_key=data_config.test_history_key,
        target_key=target_key,
        speed_index=eval_config.single_speed_index,
        snr_db=eval_config.single_snr_db,
    )
    metrics = evaluate_condition(
        model=model,
        prev_data=prev_data,
        pred_data=pred_data,
        batch_size=eval_config.batch_size,
        pred_len=model.prev_len if hasattr(model, "prev_len") else 4,
        device=device,
        se_reference_snr_db=eval_config.se_reference_snr_db,
    )
    return [
        {
            "duplex": duplex,
            "suite": "single",
            "condition": f"speed_{eval_config.single_speed_index * 10 + 10}_snr_{eval_config.single_snr_db}",
            **metrics,
        }
    ]


def run_snr_suite(
    model: torch.nn.Module,
    data_root: str,
    duplex: str,
    data_config,
    eval_config,
    device: torch.device,
) -> list[dict]:
    input_path = f"{data_root}/{data_config.test_history_path}"
    if duplex == "tdd":
        target_path = f"{data_root}/{data_config.test_tdd_target_path}"
        target_key = data_config.test_tdd_target_key
    else:
        target_path = f"{data_root}/{data_config.test_fdd_target_path}"
        target_key = data_config.test_fdd_target_key

    rows: list[dict] = []
    for snr_value in eval_config.snr_values:
        prev_data, pred_data = prepare_eval_tensors(
            input_path=input_path,
            target_path=target_path,
            input_key=data_config.test_history_key,
            target_key=target_key,
            speed_index=eval_config.snr_speed_index,
            snr_db=snr_value,
        )
        metrics = evaluate_condition(
            model=model,
            prev_data=prev_data,
            pred_data=pred_data,
            batch_size=eval_config.batch_size,
            pred_len=model.prev_len if hasattr(model, "prev_len") else 4,
            device=device,
            se_reference_snr_db=eval_config.se_reference_snr_db,
        )
        rows.append(
            {
                "duplex": duplex,
                "suite": "snr",
                "condition": f"speed_{eval_config.snr_speed_index * 10 + 10}_snr_{snr_value}",
                **metrics,
            }
        )
    return rows


def run_velocity_suite(
    model: torch.nn.Module,
    data_root: str,
    duplex: str,
    data_config,
    eval_config,
    device: torch.device,
) -> list[dict]:
    input_path = f"{data_root}/{data_config.test_history_path}"
    if duplex == "tdd":
        target_path = f"{data_root}/{data_config.test_tdd_target_path}"
        target_key = data_config.test_tdd_target_key
    else:
        target_path = f"{data_root}/{data_config.test_fdd_target_path}"
        target_key = data_config.test_fdd_target_key

    rows: list[dict] = []
    for speed_index in eval_config.velocity_indices:
        prev_data, pred_data = prepare_eval_tensors(
            input_path=input_path,
            target_path=target_path,
            input_key=data_config.test_history_key,
            target_key=target_key,
            speed_index=speed_index,
            snr_db=eval_config.velocity_snr_db,
        )
        metrics = evaluate_condition(
            model=model,
            prev_data=prev_data,
            pred_data=pred_data,
            batch_size=eval_config.batch_size,
            pred_len=model.prev_len if hasattr(model, "prev_len") else 4,
            device=device,
            se_reference_snr_db=eval_config.se_reference_snr_db,
        )
        rows.append(
            {
                "duplex": duplex,
                "suite": "velocity",
                "condition": f"speed_{speed_index * 10 + 10}_snr_{eval_config.velocity_snr_db}",
                **metrics,
            }
        )
    return rows


def suite_to_rows(
    suite_name: str,
    model: torch.nn.Module,
    data_root: str,
    duplex: str,
    data_config,
    eval_config,
    device: torch.device,
) -> list[dict]:
    mapping = {
        "single": run_single_suite,
        "snr": run_snr_suite,
        "velocity": run_velocity_suite,
    }
    if suite_name not in mapping:
        raise ValueError(f"Unsupported suite: {suite_name}")
    return mapping[suite_name](
        model=model,
        data_root=data_root,
        duplex=duplex,
        data_config=data_config,
        eval_config=eval_config,
        device=device,
    )
