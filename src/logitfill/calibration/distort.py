"""Attention-logit distortion of Key channels.

``q`` is ``[1, n_heads, T, D]`` and ``k`` is ``[1, n_kv, T, D]``. Query energy
uses positions at and after ``response_start``. ``D_c(b) = q_c e_c(b)``.
"""
from __future__ import annotations

import numpy as np
import torch

from logitfill.allocators.allocate import allocate_levels
from logitfill.quantizers.quantizer import GroupQuantizer, quantize_along_time


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


def normalize_channel_curves(
    curves: dict[str, np.ndarray],
    n_query_tokens: int,
    n_key_tokens: int,
) -> dict[str, np.ndarray]:
    """Divide summed energies by the token counts that produced them.

    ``D = q * e`` is a sum over query and key positions. Dividing by both
    counts makes a long calibration sample comparable to a short one before
    an unweighted mean across samples.
    """
    n_query = max(int(n_query_tokens), 1)
    n_key = max(int(n_key_tokens), 1)
    out = {}
    for key, value in curves.items():
        if key == "q":
            scale = float(n_query)
        elif key.startswith("e"):
            scale = float(n_key)
        elif key.startswith("d"):
            scale = float(n_query * n_key)
        else:
            continue
        out[key] = np.asarray(value, dtype=np.float64) / scale
    return out


def mean_curves(samples: list[dict[str, np.ndarray]], keys: tuple[str, ...] = ("d2", "d4", "d8", "q", "e2", "e4", "e8")) -> dict[str, np.ndarray]:
    """Equal-weight mean. Every sample contributes once, regardless of length."""
    if not samples:
        raise ValueError("mean_curves requires at least one sample")
    out = {}
    for key in keys:
        if key not in samples[0]:
            continue
        out[key] = np.mean([np.asarray(sample[key], dtype=np.float64) for sample in samples], axis=0)
    return out


def levels_for_budget(curves: dict[str, np.ndarray], extra_bits: int) -> np.ndarray:
    """One head group. ``curves`` values are ``[n_kv, head_dim]``."""
    stacked = {name: curves[name][None, ...] for name in ("d2", "d4", "d8")}
    return allocate_levels(stacked["d2"], stacked["d4"], stacked["d8"], extra_bits)[0]
