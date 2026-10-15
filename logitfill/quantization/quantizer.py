"""Asymmetric per-group fake quantization.

Matches PM-KVQ ``UntrainableQuantizer`` with ``symmetric=False`` and
``round_zeros=False``. The last dimension is split into groups. For Key,
callers transpose so that dimension is the sequence.
"""
from __future__ import annotations

import torch
from torch import nn


class GroupQuantizer(nn.Module):
    def __init__(self, n_bits: int, group_size: int = 128):
        super().__init__()
        if n_bits < 1 or n_bits > 16:
            raise ValueError(f"n_bits must be in 1..16, got {n_bits}")
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.n_bits = n_bits
        self.group_size = group_size
        self.qmin = 0
        self.qmax = (1 << n_bits) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_bits >= 16:
            return x
        if x.shape[-1] % self.group_size != 0:
            raise ValueError(
                f"last dim {x.shape[-1]} is not divisible by group_size {self.group_size}"
            )
        shape = x.shape
        dtype = x.dtype
        flat = x.reshape(-1, self.group_size)
        max_val = flat.amax(dim=1, keepdim=True)
        min_val = flat.amin(dim=1, keepdim=True)
        scales = torch.clamp((max_val - min_val) / self.qmax, min=1e-5, max=1e5)
        zeros = -min_val
        codes = torch.round((flat + zeros) / scales).clamp(self.qmin, self.qmax)
        restored = codes * scales - zeros
        return restored.reshape(shape).to(dtype)


def make_quantizers(group_size: int = 128) -> dict[int, GroupQuantizer]:
    return {bits: GroupQuantizer(bits, group_size) for bits in (2, 4, 8)}


def quantize_along_time(states: torch.Tensor, quantizer: GroupQuantizer) -> torch.Tensor:
    """``states`` is ``[B, H, T, D]``. Groups run along T."""
    return quantizer(states.transpose(-1, -2)).transpose(-1, -2)
