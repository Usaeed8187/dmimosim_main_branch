"""Loss and metric helpers."""

from __future__ import annotations

import torch
import torch.nn as nn


def nmse_value(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    power = torch.sum(target ** 2)
    mse = torch.sum((target - prediction) ** 2)
    return mse / torch.clamp(power, min=1e-12)


class NMSELoss(nn.Module):
    """Normalized mean squared error."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        value = nmse_value(prediction, target)
        if self.reduction == "sum":
            return torch.sum(value)
        return torch.mean(value)


class SpectralEfficiencyLoss(nn.Module):
    """Spectral-efficiency metric used by the original evaluation scripts."""

    def __init__(self, snr_db: float = 10.0, device: torch.device | None = None) -> None:
        super().__init__()
        self.snr_db = snr_db
        self.device = device or torch.device("cpu")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_t, n_r = prediction.shape
        h = prediction.to(self.device)
        h0 = target.to(self.device)
        if n_r != 1:
            s_real = torch.diag(torch.ones(n_r, dtype=torch.float32)).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            s_real = torch.diag(torch.ones(n_r, dtype=torch.float32)).unsqueeze(0).repeat(batch_size, 1, 1)
        s_imag = torch.zeros((batch_size, n_r, n_r), dtype=torch.float32, device=self.device)
        s = torch.complex(s_real.to(self.device), s_imag)

        matmul0 = torch.matmul(h0, s)
        fro = torch.norm(matmul0, p="fro", dim=(1, 2))
        noise_var = (torch.pow(fro, 2) / (n_t * n_r)) * pow(10, (-self.snr_db / 10))
        d = torch.adjoint(h)
        d = torch.div(d, torch.norm(d, p=2, dim=(1, 2), keepdim=True))
        d0 = torch.adjoint(h0)
        d0 = torch.div(d0, torch.norm(d0, p=2, dim=(1, 2), keepdim=True))
        matmul1 = torch.matmul(d, h0)
        matmul2 = torch.matmul(d0, h0)

        noise_var = noise_var.unsqueeze(1).unsqueeze(1)
        se = -torch.log2(torch.det(torch.div(torch.pow(torch.abs(matmul1), 2), noise_var) + s))
        se = torch.mean(se.real)

        se0 = -torch.log2(torch.det(torch.div(torch.pow(torch.abs(matmul2), 2), noise_var) + s))
        se0 = torch.mean(se0.real)
        return se, se0
