"""History payload and end-to-end KV footprint.

``target_bit`` counts only Key/Value codes on quantized history tokens.
This module adds the BF16 residual page, scale/min metadata, and packing
padding so a reported effective bit is not confused with that target.
"""
from __future__ import annotations

import numpy as np

from logitfill.allocators.allocate import LEVEL_BITS, achieved_history_bits


def _pad_channels(count: int, bits: int) -> int:
    if count <= 0:
        return 0
    width = 32 // int(bits)
    return int((count + width - 1) // width * width)


def runtime_footprint(
    levels: np.ndarray,
    seq_len: int,
    *,
    page_size: int = 128,
    k_group_size: int = 128,
    v_group_size: int = 128,
    bit_v: int = 2,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Bytes and effective bit-width of one sequence at ``seq_len`` tokens.

    The result is analytical. It does not observe allocator runtime or CUDA
    peak memory. Key channel sets are static, so their index tables are one
    copy per layer and head rather than one copy per page.
    """
    values = np.asarray(levels, dtype=np.int64)
    if values.ndim != 3:
        raise ValueError(f"levels must be [layers, kv_heads, head_dim], got {values.shape}")
    if page_size < 1 or k_group_size < 1 or v_group_size < 1:
        raise ValueError("page and group sizes must be positive")
    if page_size % k_group_size != 0:
        raise ValueError("page_size must be a multiple of k_group_size")
    n_layers, n_kv, head_dim = (int(dim) for dim in values.shape)
    if head_dim % v_group_size != 0:
        raise ValueError(f"head_dim {head_dim} is not divisible by v_group_size {v_group_size}")
    hist = int(seq_len) // int(page_size) * int(page_size)
    residual = int(seq_len) - hist
    groups = hist // int(k_group_size) if k_group_size else 0
    achieved = achieved_history_bits(values, bit_v)

    k_payload_bits = float(hist * LEVEL_BITS[values].sum())
    v_channels = n_layers * n_kv * head_dim
    if int(bit_v) >= 16:
        v_payload_bits = float(hist * 16 * v_channels)
        v_meta_bytes = 0.0
    else:
        v_payload_bits = float(hist * int(bit_v) * v_channels)
        v_groups = head_dim // int(v_group_size)
        v_meta_bytes = float(hist * n_layers * n_kv * v_groups * 2 * dtype_bytes)

    quantized = values < 3
    k_meta_bytes = float(quantized.sum() * groups * 2 * dtype_bytes)
    # One int16 channel index per Key channel, stored once for the model.
    index_bytes = float(values.size * 2)

    pad_channels = 0
    for bits, code in ((2, 0), (4, 1), (8, 2)):
        counts = (values == code).sum(axis=-1).reshape(-1)
        for count in counts:
            pad_channels += _pad_channels(int(count), bits) - int(count)
    pad_bits = float(pad_channels * hist)

    residual_bytes = float(residual * v_channels * 2 * dtype_bytes)
    total_bytes = (
        (k_payload_bits + v_payload_bits + pad_bits) / 8.0
        + k_meta_bytes
        + v_meta_bytes
        + index_bytes
        + residual_bytes
    )
    elements = float(seq_len * v_channels * 2)
    return {
        "seq_len": float(seq_len),
        "history_tokens": float(hist),
        "residual_tokens": float(residual),
        "actual_bit_k": achieved["actual_bit_k"],
        "bit_v": achieved["bit_v"],
        "actual_bit_kv": achieved["actual_bit_kv"],
        "k_payload_bytes": k_payload_bits / 8.0,
        "v_payload_bytes": v_payload_bits / 8.0,
        "packing_padding_bytes": pad_bits / 8.0,
        "metadata_bytes": k_meta_bytes + v_meta_bytes + index_bytes,
        "residual_bytes": residual_bytes,
        "total_bytes": total_bytes,
        "bytes_per_token": total_bytes / float(seq_len) if seq_len else float("nan"),
        "effective_bit": (total_bytes * 8.0 / elements) if elements else float("nan"),
    }
