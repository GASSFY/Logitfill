"""Attention-logit distortion of Key channels.

``q`` is ``[1, n_heads, T, D]`` and ``k`` is ``[1, n_kv, T, D]``. Query energy
uses positions at and after ``response_start``. ``D_c(b) = q_c e_c(b)``.
"""
from __future__ import annotations

import numpy as np
import torch

from logitfill.quantization.allocate import allocate_levels
from logitfill.quantization.quantizer import GroupQuantizer, quantize_along_time


def channel_curves(
    q: torch.Tensor,
    k: torch.Tensor,
    quantizers: dict[int, GroupQuantizer],
    response_start: int,
    n_groups: int,
) -> dict[str, np.ndarray]:
    n_kv = int(k.shape[1])
    head_dim = int(k.shape[-1])
    shape = (n_kv, head_dim)
    q_energy = np.zeros(shape, dtype=np.float64)
    errors = {}
    with torch.no_grad():
        for bits in (2, 4, 8):
            hat = quantize_along_time(k, quantizers[bits])
            errors[bits] = (hat.float() - k.float()).square()
        for head in range(n_kv):
            q0 = head * n_groups
            resp = q[0, q0 : q0 + n_groups, response_start:].float()
            q_energy[head] = resp.square().sum(dim=(0, 1)).cpu().numpy()
    curves = {"q": q_energy}
    for bits, name in ((2, "e2"), (4, "e4"), (8, "e8")):
        err = np.zeros(shape, dtype=np.float64)
        for head in range(n_kv):
            err[head] = errors[bits][0, head].sum(0).cpu().numpy()
        curves[name] = err
    curves["d2"] = q_energy * curves["e2"]
    curves["d4"] = q_energy * curves["e4"]
    curves["d8"] = q_energy * curves["e8"]
    return curves


def levels_for_budget(curves: dict[str, np.ndarray], extra_bits: int) -> np.ndarray:
    """One head group. ``curves`` values are ``[n_kv, head_dim]``."""
    stacked = {name: curves[name][None, ...] for name in ("d2", "d4", "d8")}
    return allocate_levels(stacked["d2"], stacked["d4"], stacked["d8"], extra_bits)[0]
