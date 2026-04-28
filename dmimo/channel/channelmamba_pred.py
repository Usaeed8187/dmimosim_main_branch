"""ChannelMamba predictor integration for dMIMO MU-MIMO pipeline."""

from __future__ import annotations

import os
import sys


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
        prev_len=int(getattr(cfg, "channelmamba_prev_len", 16)),
        pred_len=int(getattr(cfg, "channelmamba_pred_len", 1)),
        rb_size=int(getattr(cfg, "channelmamba_rb_size", 12)),
        max_num_rb=getattr(cfg, "channelmamba_max_num_rb", None),
        epochs=int(getattr(cfg, "channelmamba_epochs", 20)),
        batch_size=int(getattr(cfg, "channelmamba_batch_size", 128)),
        lr=float(getattr(cfg, "channelmamba_lr", 1e-3)),
        weight_decay=float(getattr(cfg, "channelmamba_weight_decay", 1e-4)),
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