"""1D-to-2D structural transforms."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSIT(nn.Module):
    """Learnable Spectral-Image Transform."""

    def __init__(self, C: int, d: int = 32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.LayerNorm(C),
            nn.Conv1d(1, d, kernel_size=1, bias=False),
            nn.Conv1d(d, d, kernel_size=5, padding=2, groups=d, bias=False),
            nn.Conv1d(d, d, kernel_size=1, bias=False),
            nn.LayerNorm([d, C]),
        )
        self.l_pos = nn.Parameter(torch.tensor(20.0))
        self.s_rbf = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.logits = nn.Parameter(torch.zeros(3))
        self.alpha_1 = nn.Parameter(torch.tensor(0.99))
        self.alpha_2 = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels = x.shape
        z = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        z = z.float()
        e = self.embed(z.unsqueeze(1)).squeeze(1).transpose(1, 2)

        def cdist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return ((a.unsqueeze(2) - b.unsqueeze(1)) ** 2).sum(-1)

        def iprod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (a.unsqueeze(2) * b.unsqueeze(1)).sum(-1)

        weights = torch.softmax(self.logits, dim=0)
        positions = torch.arange(channels, device=x.device).float()
        d_lambda = (positions[None, :, None] - positions[None, None, :]) ** 2
        k_pos = torch.exp(-d_lambda / (2 * torch.relu(self.l_pos) ** 2 + 1e-6)).repeat(batch_size, 1, 1)
        k_rbf = torch.exp(-cdist2(e, e) / (2 * torch.relu(self.s_rbf) ** 2 + 1e-6))
        k_poly = (torch.relu(self.beta) + iprod(e, e)) ** 2
        image = weights[0] * k_pos + weights[1] * k_rbf + weights[2] * k_poly

        gradient = F.pad(z[:, 1:] - z[:, :-1], (1, 0))
        gradient = gradient / (gradient.norm(dim=1, keepdim=True) + 1e-6)
        image = image + 0.1 * (gradient.unsqueeze(2) @ gradient.unsqueeze(1))
        image = (image - image.mean(dim=(1, 2), keepdim=True)) / (image.std(dim=(1, 2), keepdim=True) + 1e-6)
        return image.unsqueeze(1), weights

    def compute_losses(self, image: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        row_mean = image.mean(dim=2).squeeze(1)
        col_mean = image.mean(dim=3).squeeze(1)
        consistency_loss = F.l1_loss(row_mean, x_norm) + F.l1_loss(col_mean, x_norm)

        x_hat = (row_mean + col_mean) / 2
        dot = torch.sum(x * x_hat, dim=1)
        denom = x.norm(dim=1) * x_hat.norm(dim=1) + 1e-6
        shape_loss = (1 - dot / denom).mean()
        return self.alpha_1 * consistency_loss + self.alpha_2 * shape_loss


class GAFTransform(nn.Module):
    """Gramian Angular Field transform."""

    def __init__(self, clamp_eps: float = 1e-6):
        super().__init__()
        self.clamp_eps = clamp_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x - x.min(dim=1, keepdim=True).values
        x1 = x0 / (x0.max(dim=1, keepdim=True).values + 1e-6)
        scaled = torch.clamp(x1 * 2.0 - 1.0, -1.0 + self.clamp_eps, 1.0 - self.clamp_eps)
        phi = torch.arccos(scaled)
        image = torch.cos(phi.unsqueeze(2) + phi.unsqueeze(1))
        image = (image - image.mean(dim=(1, 2), keepdim=True)) / (image.std(dim=(1, 2), keepdim=True) + 1e-6)
        return image.unsqueeze(1)


class MTFTransform(nn.Module):
    """Markov Transition Field transform."""

    def __init__(self, Q: int = 16):
        super().__init__()
        self.Q = Q

    @torch.no_grad()
    def _markov_p(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, _ = states.shape
        p_all = torch.zeros(batch_size, self.Q, self.Q, device=states.device)
        for b in range(batch_size):
            matrix = torch.zeros(self.Q, self.Q, device=states.device)
            i = states[b, :-1]
            j = states[b, 1:]
            matrix.index_put_((i, j), torch.ones_like(i, dtype=matrix.dtype), accumulate=True)
            p_all[b] = matrix / (matrix.sum(dim=1, keepdim=True) + 1e-6)
        return p_all

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x - x.min(dim=1, keepdim=True).values
        x1 = x0 / (x0.max(dim=1, keepdim=True).values + 1e-6)
        bins = torch.clamp((x1 * self.Q).long(), 0, self.Q - 1)
        transition = self._markov_p(bins)
        image = torch.stack([transition[b][bins[b].unsqueeze(1), bins[b].unsqueeze(0)] for b in range(x.size(0))])
        image = (image - image.mean(dim=(1, 2), keepdim=True)) / (image.std(dim=(1, 2), keepdim=True) + 1e-6)
        return image.unsqueeze(1)


class RPTransform(nn.Module):
    """Gaussian-kernel recurrence plot transform."""

    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        dist2 = (x_n.unsqueeze(2) - x_n.unsqueeze(1)) ** 2
        image = torch.exp(-dist2 / (2 * (self.sigma**2) + 1e-6))
        image = (image - image.mean(dim=(1, 2), keepdim=True)) / (image.std(dim=(1, 2), keepdim=True) + 1e-6)
        return image.unsqueeze(1)

