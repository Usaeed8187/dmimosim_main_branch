"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    d_in: int = 96
    prev_len: int = 16
    pred_len: int = 4
    d_model: int = 512
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    td_headdim: int = 64
    ngroups: int = 1
    num_td_layers: int = 2
    num_bimamba_layers: int = 3
    num_ffn_layers: int = 1
    d_ff_ratio: int = 4
    d_tf: int = 128
    n_head: int = 4
    num_transformer_layers: int = 1
    fusion_type: str = "concat"
    dropout: float = 0.1
    ssm_backend: str = "mamba2"


@dataclass
class DataConfig:
    train_history_path: str = "train/H_U_his_train.mat"
    train_tdd_target_path: str = "train/H_U_pre_train.mat"
    train_fdd_target_path: str = "train/H_D_pre_train.mat"
    test_history_path: str = "test/H_U_his_test.mat"
    test_tdd_target_path: str = "test/H_U_pre_test.mat"
    test_fdd_target_path: str = "test/H_D_pre_test.mat"
    train_history_key: str = "H_U_his_train"
    train_tdd_target_key: str = "H_U_pre_train"
    train_fdd_target_key: str = "H_D_pre_train"
    test_history_key: str = "H_U_his_test"
    test_tdd_target_key: str = "H_U_pre_test"
    test_fdd_target_key: str = "H_D_pre_test"
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    antenna_group_size: int = 32
    train_noise_min_snr_db: float = 5.0
    train_noise_max_snr_db: float = 20.0


@dataclass
class TrainConfig:
    lr: float = 1e-4
    epochs: int = 500
    batch_size: int = 128
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eta_min: float = 1e-7
    grad_clip: float = 1.0
    seed: int = 1234
    max_train_batches: int | None = None
    max_val_batches: int | None = None


@dataclass
class EvalConfig:
    batch_size: int = 64
    se_reference_snr_db: float = 10.0
    single_speed_index: int = 5
    single_snr_db: float = 10.0
    snr_speed_index: int = 5
    snr_values: list[int] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    velocity_indices: list[int] = field(default_factory=lambda: list(range(10)))
    velocity_snr_db: float = 18.0


@dataclass
class RuntimeConfig:
    device: str = "auto"
    output_root: str = "outputs"
    run_name: str | None = None
    num_workers: int = 4
    use_data_parallel: bool = True
    save_last: bool = True
    log_every: int = 20


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for item in fields(instance):
        if item.name not in updates:
            continue
        current = getattr(instance, item.name)
        incoming = updates[item.name]
        if is_dataclass(current) and isinstance(incoming, dict):
            _update_dataclass(current, incoming)
        else:
            setattr(instance, item.name, incoming)
    return instance


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is None:
        return config
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config file must contain a mapping: {config_path}")
    return _update_dataclass(config, raw)
