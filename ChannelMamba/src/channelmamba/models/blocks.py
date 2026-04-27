"""Reusable building blocks for ChannelMamba."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

_MAMBA2 = None
_MAMBA2_ERROR = None
_MAMBA1 = None
_MAMBA1_ERROR = None


def _load_mamba2():
    global _MAMBA2, _MAMBA2_ERROR
    if _MAMBA2 is not None or _MAMBA2_ERROR is not None:
        return _MAMBA2
    try:  # pragma: no cover - optional dependency
        from mamba_ssm.modules.mamba2 import Mamba2 as imported_mamba2
    except Exception as exc:  # pragma: no cover - optional dependency
        _MAMBA2_ERROR = exc
        return None
    _MAMBA2 = imported_mamba2
    return _MAMBA2


def _load_mamba1():
    global _MAMBA1, _MAMBA1_ERROR
    if _MAMBA1 is not None or _MAMBA1_ERROR is not None:
        return _MAMBA1
    try:  # pragma: no cover - optional dependency
        from mamba_ssm.modules.mamba_simple import Mamba as imported_mamba1
    except Exception as exc:  # pragma: no cover - optional dependency
        _MAMBA1_ERROR = exc
        return None
    _MAMBA1 = imported_mamba1
    return _MAMBA1


class LinearSequenceMixer(nn.Module):
    """Fallback mixer used for smoke tests when Mamba is unavailable."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def build_sequence_mixer(
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    backend: str,
    headdim: int | None = None,
    ngroups: int = 1,
    dropout: float = 0.1,
) -> nn.Module:
    if backend == "linear":
        return LinearSequenceMixer(d_model=d_model, dropout=dropout)
    if backend == "mamba2":
        mamba2 = _load_mamba2()
        if mamba2 is None:
            raise ImportError(
                "mamba_ssm.modules.mamba2.Mamba2 is required for ssm_backend='mamba2'. "
                f"Original import error: {_MAMBA2_ERROR!r}"
            )
        return mamba2(d_model=d_model, d_state=d_state, headdim=headdim or d_model, ngroups=ngroups)
    if backend == "mamba":
        mamba1 = _load_mamba1()
        if mamba1 is None:
            raise ImportError(
                "mamba_ssm.modules.mamba_simple.Mamba is required for ssm_backend='mamba'. "
                f"Original import error: {_MAMBA1_ERROR!r}"
            )
        return mamba1(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    if backend == "auto":
        mamba2 = _load_mamba2()
        if mamba2 is not None:
            return mamba2(d_model=d_model, d_state=d_state, headdim=headdim or d_model, ngroups=ngroups)
        mamba1 = _load_mamba1()
        if mamba1 is not None:
            warnings.warn("Mamba-2 is unavailable; falling back to Mamba-1.", stacklevel=2)
            return mamba1(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        raise ImportError("Neither Mamba2 nor Mamba is available; install mamba-ssm or use ssm_backend='linear'.")
    raise ValueError(f"Unsupported ssm backend: {backend}")


class SharedTDResidualBlock(nn.Module):
    """Shared temporal-dynamics block used in both frequency and delay branches."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        backend: str,
        headdim: int,
        ngroups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = build_sequence_mixer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            backend=backend,
            headdim=headdim,
            ngroups=ngroups,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.mixer(self.norm(inputs))
        return residual + self.dropout(outputs)


class BidirectionalFeatureBlock(nn.Module):
    """BiMamba-like block that scans along the feature dimension."""

    def __init__(
        self,
        prev_len: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        backend: str,
        ngroups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.prev_len = prev_len
        self.norm = nn.LayerNorm(d_model)
        self.forward_mixer = build_sequence_mixer(
            d_model=prev_len,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            backend=backend,
            headdim=prev_len,
            ngroups=ngroups,
            dropout=dropout,
        )
        self.backward_mixer = build_sequence_mixer(
            d_model=prev_len,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            backend=backend,
            headdim=prev_len,
            ngroups=ngroups,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        normalized = self.norm(inputs)
        transposed = normalized.permute(0, 2, 1)

        forward_outputs = self.forward_mixer(transposed)
        backward_inputs = torch.flip(transposed, dims=[1])
        backward_outputs = torch.flip(self.backward_mixer(backward_inputs), dims=[1])

        merged = (forward_outputs + backward_outputs).permute(0, 2, 1)
        return self.output_norm(residual + self.dropout(merged))


class FeedForwardBlock(nn.Module):
    """Residual feed-forward block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.linear1(self.norm(inputs))
        outputs = self.activation(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.linear2(outputs)
        outputs = self.dropout2(outputs)
        return residual + outputs
