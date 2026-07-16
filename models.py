"""Backward-compatible model exports.

The model implementation is split across focused modules:
- augment.py
- spectral_branch.py
- transforms.py
- image_branch.py
- cross_attention.py
- dual_stream.py
"""

from .augment import SpectralAugment
from .cross_attention import DynamicCrossAttention
from .dual_stream import DualStreamNIRNet
from .image_branch import ImageBranch
from .spectral_branch import Attention, SpectrumModel
from .transforms import GAFTransform, LSIT, MTFTransform, RPTransform

__all__ = [
    "Attention",
    "SpectralAugment",
    "SpectrumModel",
    "LSIT",
    "GAFTransform",
    "MTFTransform",
    "RPTransform",
    "ImageBranch",
    "DynamicCrossAttention",
    "DualStreamNIRNet",
]
