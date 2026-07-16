"""Dual-stream classifier with structural transform, DCA, and gated fusion."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention import DynamicCrossAttention
from .image_branch import ImageBranch
from .spectral_branch import SpectrumModel
from .transforms import GAFTransform, LSIT, MTFTransform, RPTransform


class DualStreamNIRNet(nn.Module):
    """DCA -> DGF + FAL dual-stream classifier."""

    def __init__(self, input_shape: int = 256, num_classes: int = 4, transform: str = "lsit", mtf_Q: int = 16, rp_sigma: float = 0.1):
        super().__init__()
        self.spec_branch = SpectrumModel(input_shape)
        self.transform = transform.lower()
        if self.transform == "lsit":
            self.transformer = LSIT(C=input_shape)
            self.use_lsit_loss = True
        elif self.transform == "gaf":
            self.transformer = GAFTransform()
            self.use_lsit_loss = False
        elif self.transform == "mtf":
            self.transformer = MTFTransform(Q=mtf_Q)
            self.use_lsit_loss = False
        elif self.transform == "rp":
            self.transformer = RPTransform(sigma=rp_sigma)
            self.use_lsit_loss = False
        else:
            raise ValueError(f"Unknown transform: {transform}")

        self.img_branch = ImageBranch()
        self.dca = DynamicCrossAttention(d_model=128, num_heads=4, n_tokens=8, use_topk=True, topk_ratio=0.5, dropout=0.1)
        self.gate = nn.Sequential(nn.Linear(128 * 3, 128), nn.ReLU(inplace=True), nn.Linear(128, 3), nn.Tanh())
        self.proj_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 32))
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, num_classes)

    @staticmethod
    def _cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        a_n = a / (a.norm(dim=1, keepdim=True) + eps)
        b_n = b / (b.norm(dim=1, keepdim=True) + eps)
        return (a_n * b_n).sum(dim=1)

    def compute_fusion_losses(
        self,
        spec_feat: torch.Tensor,
        img_feat: torch.Tensor,
        cross_feat: torch.Tensor,
        fused: torch.Tensor,
        gate_weights: torch.Tensor,
        lambda_stage1: float = 0.1,
        lambda_stage2: float = 0.1,
        lambda_gate: float = 1e-3,
        lambda_decor: float = 1e-3,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cos_cs = self._cos_sim(cross_feat, spec_feat)
        cos_ci = self._cos_sim(cross_feat, img_feat)
        stage1_loss = (1.0 - cos_cs).mean() + (1.0 - cos_ci).mean()

        z_f = self.proj_head(fused)
        z_c = self.proj_head(cross_feat)
        stage2_loss = F.mse_loss(z_f, z_c) + (1.0 - self._cos_sim(fused, cross_feat).mean())

        gw = gate_weights.clamp(1e-8, 1.0)
        entropy = -(gw * gw.log()).sum(dim=1).mean()
        gate_loss = -entropy
        decor_loss = (self._cos_sim(spec_feat, img_feat) ** 2).mean()

        total = lambda_stage1 * stage1_loss + lambda_stage2 * stage2_loss + lambda_gate * gate_loss + lambda_decor * decor_loss
        stats = {
            "L_stage1": stage1_loss.item(),
            "L_stage2": stage2_loss.item(),
            "L_gate": gate_loss.item(),
            "L_decor": decor_loss.item(),
            "cos_cs": cos_cs.mean().item(),
            "cos_ci": cos_ci.mean().item(),
            "cos_fc": self._cos_sim(fused, cross_feat).mean().item(),
            "H_gate": entropy.item(),
        }
        return total, stats

    def forward(self, x: torch.Tensor, return_loss: bool = False):
        spec_feat = self.spec_branch(x)
        if self.transform == "lsit":
            image, weights = self.transformer(x)
        else:
            image = self.transformer(x)
            weights = torch.zeros(3, device=x.device, dtype=x.dtype)
        img_feat = self.img_branch(image)
        cross_feat, tau_mean = self.dca(spec_feat, img_feat)

        fusion_input = torch.cat([spec_feat, img_feat, cross_feat], dim=1)
        gate_weights = self.gate(fusion_input)
        alpha, beta, gamma = gate_weights[:, 0:1], gate_weights[:, 1:2], gate_weights[:, 2:3]
        fused = alpha * spec_feat + beta * img_feat + gamma * cross_feat
        logits = self.fc2(F.relu(self.fc1(fused)))

        if not return_loss:
            return logits, weights

        if self.use_lsit_loss:
            lsit_loss = self.transformer.compute_losses(image, x)
        else:
            lsit_loss = torch.zeros((), device=x.device, dtype=spec_feat.dtype)
        fal_loss, fal_stats = self.compute_fusion_losses(spec_feat, img_feat, cross_feat, fused, gate_weights)
        gate_means = (alpha.mean().item(), beta.mean().item(), gamma.mean().item())
        return logits, lsit_loss, fal_loss, tau_mean, gate_means, fal_stats, weights

