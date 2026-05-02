"""ChannelMamba predictor integration for dMIMO MU-MIMO pipeline."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
import numpy as np

def _ensure_channelmamba_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    channelmamba_src = os.path.join(repo_root, "ChannelMamba", "src")
    if channelmamba_src not in sys.path:
        sys.path.append(channelmamba_src)


_ensure_channelmamba_on_path()

from channelmamba.dmimo_bridge import DMIMOChannelMambaConfig, DMIMOChannelMambaPredictor


def build_channelmamba_predictor(cfg) -> DMIMOChannelMambaPredictor:
    channelmamba_mode = str(getattr(cfg, "channelmamba_mode", "train")).lower()
    cm_cfg = DMIMOChannelMambaConfig(
        prev_len=int(getattr(cfg, "channelmamba_prev_len", 8)),
        pred_len=int(getattr(cfg, "channelmamba_pred_len", 1)),
        rb_size=int(getattr(cfg, "channelmamba_rb_size", 12)),
        max_num_rb=getattr(cfg, "channelmamba_max_num_rb", None),
        epochs=int(getattr(cfg, "channelmamba_epochs", 20)),
        batch_size=int(getattr(cfg, "channelmamba_batch_size", 128)),
        lr=float(getattr(cfg, "channelmamba_lr", 1e-3)),
        weight_decay=float(getattr(cfg, "channelmamba_weight_decay", 1e-4)),
        grad_clip=float(getattr(cfg, "channelmamba_grad_clip", 0.0)),
        seed=int(getattr(cfg, "channelmamba_seed", 1234)),
        d_model=int(getattr(cfg, "channelmamba_d_model", 256)),
        d_state=int(getattr(cfg, "channelmamba_d_state", 16)),
        d_conv=int(getattr(cfg, "channelmamba_d_conv", 4)),
        expand=int(getattr(cfg, "channelmamba_expand", 2)),
        td_headdim=int(getattr(cfg, "channelmamba_td_headdim", 64)),
        ngroups=int(getattr(cfg, "channelmamba_ngroups", 1)),
        num_td_layers=int(getattr(cfg, "channelmamba_num_td_layers", 2)),
        num_bimamba_layers=int(getattr(cfg, "channelmamba_num_bimamba_layers", 2)),
        num_ffn_layers=int(getattr(cfg, "channelmamba_num_ffn_layers", 1)),
        d_ff_ratio=int(getattr(cfg, "channelmamba_d_ff_ratio", 4)),
        d_tf=int(getattr(cfg, "channelmamba_d_tf", 128)),
        n_head=int(getattr(cfg, "channelmamba_n_head", 4)),
        num_transformer_layers=int(getattr(cfg, "channelmamba_num_transformer_layers", 1)),
        fusion_type=str(getattr(cfg, "channelmamba_fusion_type", "concat")),
        dropout=float(getattr(cfg, "channelmamba_dropout", 0.1)),
        ssm_backend=str(getattr(cfg, "channelmamba_ssm_backend", "auto")),
        enable_tf32=bool(getattr(cfg, "channelmamba_enable_tf32", True)),
        dataloader_num_workers=int(getattr(cfg, "channelmamba_dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(getattr(cfg, "channelmamba_dataloader_pin_memory", True)),
        dataloader_persistent_workers=bool(getattr(cfg, "channelmamba_dataloader_persistent_workers", True)),
        dataloader_prefetch_factor=getattr(cfg, "channelmamba_dataloader_prefetch_factor", 2),
        torch_compile=bool(getattr(cfg, "channelmamba_torch_compile", False)),
        torch_compile_mode=getattr(cfg, "channelmamba_torch_compile_mode", None),
        checkpoint_path=getattr(cfg, "channelmamba_checkpoint", None),
        freeze_loaded_checkpoint=channelmamba_mode == "eval",
        checkpoint_metadata=getattr(cfg, "channelmamba_checkpoint_metadata", None),
        allow_mismatch_reset=bool(getattr(cfg, "channelmamba_allow_mismatch_reset", False)),
    )
    return DMIMOChannelMambaPredictor(cm_cfg)


def predict_all_links_with_channelmamba_simple(
    h_freq_csi_history,
    channelmamba_predictor: DMIMOChannelMambaPredictor,
    ns3cfg,
    num_bs_ant=4,
    num_ue_ant=2,
):
    return channelmamba_predictor.predict_all_links(
        h_freq_csi_history,
        ns3cfg=ns3cfg,
        num_bs_ant=num_bs_ant,
        num_ue_ant=num_ue_ant,
    )

def _pair_indices(node_idx: int, num_bs_ant: int, num_ue_ant: int):
    if node_idx == 0:
        return list(range(0, num_bs_ant))
    start = num_bs_ant + (node_idx - 1) * num_ue_ant
    end = num_bs_ant + node_idx * num_ue_ant
    return list(range(start, end))


def _pair_checkpoint_path(base_path: str | None, tx_node_idx: int, rx_node_idx: int) -> str | None:
    if not base_path:
        return None
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".pt"
    return f"{root}__tx{tx_node_idx}_rx{rx_node_idx}{ext}"

def _shared_group_checkpoint_path(base_path: str | None, num_tx_ant: int, num_rx_ant: int) -> str | None:
    if not base_path:
        return None
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".pt"
    return f"{root}__shared_txant{num_tx_ant}_rxant{num_rx_ant}{ext}"

def build_channelmamba_predictors_simple(
    pooled_h_hist_per_drop,
    cfg,
    ns3cfg,
    num_bs_ant=4,
    num_ue_ant=2,
):
    channelmamba_mode = str(getattr(cfg, "channelmamba_mode", "train")).lower()
    base_metadata = deepcopy(getattr(cfg, "channelmamba_checkpoint_metadata", {}) or {})
    base_checkpoint = getattr(cfg, "channelmamba_checkpoint", None)
    predictors = {}
    pair_groups = {}

    for tx_node_idx in range(ns3cfg.num_txue_sel + 1):
        for rx_node_idx in range(ns3cfg.num_rxue_sel + 1):
            tx_ant_idx = _pair_indices(tx_node_idx, num_bs_ant=num_bs_ant, num_ue_ant=num_ue_ant)
            rx_ant_idx = _pair_indices(rx_node_idx, num_bs_ant=num_bs_ant, num_ue_ant=num_ue_ant)

            group_key = (int(len(tx_ant_idx)), int(len(rx_ant_idx)))
            pair_groups.setdefault(group_key, []).append((tx_node_idx, rx_node_idx, tx_ant_idx, rx_ant_idx))

    for (num_tx_ant, num_rx_ant), group_pairs in pair_groups.items():
        pooled_group_histories = []
        pair_ids = []
        for tx_node_idx, rx_node_idx, tx_ant_idx, rx_ant_idx in group_pairs:
            pair_ids.append((int(tx_node_idx), int(rx_node_idx)))
            for drop_hist in pooled_h_hist_per_drop:
                curr_h = drop_hist[:, :, :, rx_ant_idx, :, ...]
                curr_h = curr_h[:, :, :, :, :, tx_ant_idx, ...]
                pooled_group_histories.append(curr_h)

        group_metadata = deepcopy(base_metadata)
        group_metadata["num_tx_ant"] = int(num_tx_ant)
        group_metadata["num_rx_ant"] = int(num_rx_ant)
        group_metadata["shared_pair_ids"] = pair_ids

        predictor_cfg = DMIMOChannelMambaConfig(
            prev_len=int(getattr(cfg, "channelmamba_prev_len", 8)),
            pred_len=int(getattr(cfg, "channelmamba_pred_len", 1)),
            rb_size=int(getattr(cfg, "channelmamba_rb_size", 12)),
            max_num_rb=getattr(cfg, "channelmamba_max_num_rb", None),
            epochs=int(getattr(cfg, "channelmamba_epochs", 20)),
            batch_size=int(getattr(cfg, "channelmamba_batch_size", 128)),
            lr=float(getattr(cfg, "channelmamba_lr", 1e-3)),
            weight_decay=float(getattr(cfg, "channelmamba_weight_decay", 1e-4)),
            grad_clip=float(getattr(cfg, "channelmamba_grad_clip", 0.0)),
            seed=int(getattr(cfg, "channelmamba_seed", 1234)),
            d_model=int(getattr(cfg, "channelmamba_d_model", 256)),
            d_state=int(getattr(cfg, "channelmamba_d_state", 16)),
            d_conv=int(getattr(cfg, "channelmamba_d_conv", 4)),
            expand=int(getattr(cfg, "channelmamba_expand", 2)),
            td_headdim=int(getattr(cfg, "channelmamba_td_headdim", 64)),
            ngroups=int(getattr(cfg, "channelmamba_ngroups", 1)),
            num_td_layers=int(getattr(cfg, "channelmamba_num_td_layers", 2)),
            num_bimamba_layers=int(getattr(cfg, "channelmamba_num_bimamba_layers", 2)),
            num_ffn_layers=int(getattr(cfg, "channelmamba_num_ffn_layers", 1)),
            d_ff_ratio=int(getattr(cfg, "channelmamba_d_ff_ratio", 4)),
            d_tf=int(getattr(cfg, "channelmamba_d_tf", 128)),
            n_head=int(getattr(cfg, "channelmamba_n_head", 4)),
            num_transformer_layers=int(getattr(cfg, "channelmamba_num_transformer_layers", 1)),
            fusion_type=str(getattr(cfg, "channelmamba_fusion_type", "concat")),
            dropout=float(getattr(cfg, "channelmamba_dropout", 0.1)),
            ssm_backend=str(getattr(cfg, "channelmamba_ssm_backend", "auto")),
            device=str(getattr(cfg, "channelmamba_device", "cuda")),
            enable_tf32=bool(getattr(cfg, "channelmamba_enable_tf32", True)),
            dataloader_num_workers=int(getattr(cfg, "channelmamba_dataloader_num_workers", 4)),
            dataloader_pin_memory=bool(getattr(cfg, "channelmamba_dataloader_pin_memory", True)),
            dataloader_persistent_workers=bool(getattr(cfg, "channelmamba_dataloader_persistent_workers", True)),
            dataloader_prefetch_factor=getattr(cfg, "channelmamba_dataloader_prefetch_factor", 2),
            torch_compile=bool(getattr(cfg, "channelmamba_torch_compile", False)),
            torch_compile_mode=getattr(cfg, "channelmamba_torch_compile_mode", None),
            checkpoint_path=_shared_group_checkpoint_path(base_checkpoint, num_tx_ant=num_tx_ant, num_rx_ant=num_rx_ant),
            freeze_loaded_checkpoint=channelmamba_mode == "eval",
            checkpoint_metadata=group_metadata,
            allow_mismatch_reset=bool(getattr(cfg, "channelmamba_allow_mismatch_reset", False)),
        )

        predictor = DMIMOChannelMambaPredictor(predictor_cfg)
        if channelmamba_mode == "eval":
            predictor.load_checkpoint_only()
        else:
            predictor.fit_offline_pair(pooled_group_histories)

        for tx_node_idx, rx_node_idx, _tx_ant_idx, _rx_ant_idx in group_pairs:
            predictors[(tx_node_idx, rx_node_idx)] = predictor

    return predictors


def predict_all_links_with_channelmamba_per_pair(
    h_freq_csi_history,
    channelmamba_predictors,
    ns3cfg,
    num_bs_ant=4,
    num_ue_ant=2,
):
    h_freq_csi = np.zeros(h_freq_csi_history[0, ...].shape, dtype=h_freq_csi_history.dtype)
    for tx_node_idx in range(ns3cfg.num_txue_sel + 1):
        for rx_node_idx in range(ns3cfg.num_rxue_sel + 1):
            predictor = channelmamba_predictors[(tx_node_idx, rx_node_idx)]
            tx_ant_idx = _pair_indices(tx_node_idx, num_bs_ant=num_bs_ant, num_ue_ant=num_ue_ant)
            rx_ant_idx = _pair_indices(rx_node_idx, num_bs_ant=num_bs_ant, num_ue_ant=num_ue_ant)

            curr_h = h_freq_csi_history[:, :, :, rx_ant_idx, :, ...]
            curr_h = curr_h[:, :, :, :, :, tx_ant_idx, ...]
            tmp = predictor.predict_pair(curr_h)

            rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
            h_freq_csi[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

    return h_freq_csi