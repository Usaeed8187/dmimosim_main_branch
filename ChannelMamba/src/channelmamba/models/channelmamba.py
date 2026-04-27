"""Official ChannelMamba model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from .blocks import BidirectionalFeatureBlock, FeedForwardBlock, SharedTDResidualBlock


class ChannelMamba(nn.Module):
    """ChannelMamba reference implementation aligned with the paper design."""

    def __init__(
        self,
        d_in: int = 96,
        prev_len: int = 16,
        pred_len: int = 4,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        td_headdim: int = 64,
        ngroups: int = 1,
        num_td_layers: int = 2,
        num_bimamba_layers: int = 3,
        num_ffn_layers: int = 1,
        d_ff_ratio: int = 4,
        d_tf: int = 128,
        n_head: int = 4,
        num_transformer_layers: int = 1,
        fusion_type: str = "concat",
        dropout: float = 0.1,
        ssm_backend: str = "mamba2",
    ) -> None:
        super().__init__()
        if d_in != 96:
            raise ValueError("ChannelMamba expects d_in=96 based on the released preprocessing pipeline.")
        if fusion_type not in {"concat", "add"}:
            raise ValueError("fusion_type must be 'concat' or 'add'.")
        if d_tf % n_head != 0:
            raise ValueError(f"d_tf ({d_tf}) must be divisible by n_head ({n_head}).")
        if d_model % td_headdim != 0 and ssm_backend in {"mamba2", "auto"}:
            raise ValueError(f"d_model ({d_model}) must be divisible by td_headdim ({td_headdim}).")

        self.d_in = d_in
        self.prev_len = prev_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.fusion_type = fusion_type

        self.embedding_freq = nn.Linear(d_in, d_model)
        self.embedding_delay = nn.Linear(d_in, d_model)
        self.shared_embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        self.shared_td_blocks = nn.ModuleList(
            [
                SharedTDResidualBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    backend=ssm_backend,
                    headdim=td_headdim,
                    ngroups=ngroups,
                    dropout=dropout,
                )
                for _ in range(num_td_layers)
            ]
        )

        if fusion_type == "concat":
            self.fusion = nn.Linear(d_model * 2, d_model)
        else:
            self.fusion = nn.Identity()
        self.fusion_norm = nn.LayerNorm(d_model)

        self.bimamba_blocks = nn.ModuleList(
            [
                BidirectionalFeatureBlock(
                    prev_len=prev_len,
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    backend=ssm_backend,
                    ngroups=ngroups,
                    dropout=dropout,
                )
                for _ in range(num_bimamba_layers)
            ]
        )
        self.ffn_blocks = nn.ModuleList(
            [FeedForwardBlock(d_model=d_model, d_ff=d_model * d_ff_ratio, dropout=dropout) for _ in range(num_ffn_layers)]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_tf,
            nhead=n_head,
            dim_feedforward=d_tf * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_input = nn.Linear(d_model, d_tf)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.transformer_norm = nn.LayerNorm(d_tf)
        self.output_projection = nn.Linear(d_tf, d_in)
        self.time_projection = nn.Linear(prev_len, pred_len)

    @classmethod
    def from_config(cls, config) -> "ChannelMamba":
        return cls(**config.__dict__)

    def _compute_delay_domain(self, frequency_inputs: torch.Tensor) -> torch.Tensor:
        batch, steps, features = frequency_inputs.shape
        if features != self.d_in:
            raise ValueError(f"Expected input feature dimension {self.d_in}, but received {features}.")
        real_imag = rearrange(frequency_inputs, "b t (k two) -> b t k two", two=2)
        frequency_complex = torch.complex(real_imag[..., 0], real_imag[..., 1])
        delay_complex = torch.fft.ifft(frequency_complex, dim=-1)
        delay_real_imag = torch.stack([delay_complex.real, delay_complex.imag], dim=-1)
        return delay_real_imag.reshape(batch, steps, features)

    def _apply_shared_td_path(self, inputs: torch.Tensor, embedding: nn.Linear) -> torch.Tensor:
        outputs = embedding(inputs)
        outputs = self.shared_embedding_norm(outputs)
        outputs = self.embedding_dropout(outputs)
        for block in self.shared_td_blocks:
            outputs = block(outputs)
        return outputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, prev_len, features = inputs.shape
        if prev_len != self.prev_len:
            raise ValueError(f"Expected prev_len={self.prev_len}, but received {prev_len}.")
        if features != self.d_in:
            raise ValueError(f"Expected d_in={self.d_in}, but received {features}.")

        delay_inputs = self._compute_delay_domain(inputs)
        freq_features = self._apply_shared_td_path(inputs, self.embedding_freq)
        delay_features = self._apply_shared_td_path(delay_inputs, self.embedding_delay)

        if self.fusion_type == "concat":
            fused = self.fusion(torch.cat([freq_features, delay_features], dim=-1))
        else:
            fused = freq_features + delay_features
        fused = self.fusion_norm(fused)

        outputs = fused
        for block in self.bimamba_blocks:
            outputs = block(outputs)
        for block in self.ffn_blocks:
            outputs = block(outputs)

        outputs = self.transformer_input(outputs)
        outputs = self.transformer(outputs)
        outputs = self.transformer_norm(outputs)
        outputs = self.output_projection(outputs)
        outputs = self.time_projection(outputs.permute(0, 2, 1)).permute(0, 2, 1)
        return outputs
