"""Spectral data augmentation utilities."""

from __future__ import annotations

from typing import Tuple

import torch


class SpectralAugment:
    """Lightweight spectral augmentation: jitter, scale, shift, and cutout."""

    def __init__(
        self,
        p_jitter: float = 0.8,
        p_scale: float = 0.8,
        p_shift: float = 0.5,
        p_cutout: float = 0.5,
        jitter_sigma: float = 0.01,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shift_sigma: float = 0.01,
        cutout_max_width: int = 20,
    ):
        self.p_jitter = p_jitter
        self.p_scale = p_scale
        self.p_shift = p_shift
        self.p_cutout = p_cutout
        self.jitter_sigma = jitter_sigma
        self.scale_range = scale_range
        self.shift_sigma = shift_sigma
        self.cutout_max_width = cutout_max_width

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("SpectralAugment only supports torch.Tensor input.")

        original_dim = x.dim()
        if original_dim == 1:
            x = x.unsqueeze(0)
            squeeze_back = "1d"
        elif original_dim == 3:
            if x.size(1) != 1:
                raise ValueError("3D spectral input must be shaped [B, 1, L].")
            x = x.squeeze(1)
            squeeze_back = "b1l"
        else:
            squeeze_back = "bl"

        batch_size, length = x.shape
        out = x.clone()

        if torch.rand(1, device=out.device) < self.p_jitter:
            out = out + torch.randn_like(out) * self.jitter_sigma
        if torch.rand(1, device=out.device) < self.p_scale:
            scale = torch.empty(batch_size, 1, device=out.device).uniform_(*self.scale_range)
            out = out * scale
        if torch.rand(1, device=out.device) < self.p_shift:
            out = out + torch.randn(batch_size, 1, device=out.device) * self.shift_sigma
        if torch.rand(1, device=out.device) < self.p_cutout and length > 1:
            widths = torch.randint(1, min(self.cutout_max_width, length) + 1, (batch_size,), device=out.device)
            starts = torch.randint(0, length - widths.max() + 1, (batch_size,), device=out.device)
            for i in range(batch_size):
                out[i, starts[i] : starts[i] + widths[i]] = out[i].mean()

        if squeeze_back == "1d":
            return out.squeeze(0)
        if squeeze_back == "b1l":
            return out.unsqueeze(1)
        return out

