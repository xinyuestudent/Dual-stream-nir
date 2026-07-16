"""1D spectral branch modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Channel attention used by the 1D spectral branch."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.softmax(self.conv(x))
        return x * att


class SpectrumModel(nn.Module):
    """1D convolution + attention + Transformer branch."""

    def __init__(self, input_shape: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.att = Attention(16)

        self.embedding_dim = 16
        self.pos_encoding = nn.Parameter(torch.randn(1, input_shape, self.embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.LazyLinear(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.att(x)

        x = x.permute(0, 2, 1)
        seq_len = x.size(1)
        if seq_len > self.pos_encoding.size(1):
            raise ValueError(f"Input length {seq_len} exceeds configured input_shape {self.pos_encoding.size(1)}.")
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = x.flatten(1)
        return F.relu(self.fc1(x))

