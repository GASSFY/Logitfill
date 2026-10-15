"""Reverse water-filling for Key channels.

Levels are 0=int2, 1=int4, 2=int8, 3=fp16. Each head starts at int2.
Extra bits are counted above that floor. One purchase upgrades a channel
by a single step: 2→4 costs 2, 4→8 costs 4, 8→16 costs 8.
"""
from __future__ import annotations

import numpy as np

LEVEL_BITS = np.array([2, 4, 8, 16], dtype=np.int64)


def extra_bits(target_bit: float, n_channels: int = 128) -> int:
    """Spare bits per head above an int2 floor."""
    spare = (float(target_bit) - 2.0) * n_channels
    if spare < 0:
        raise ValueError(f"target_bit {target_bit} is below the int2 floor")
    return int(round(spare))


def waterfill(
    d2: np.ndarray,
    d4: np.ndarray,
    d8: np.ndarray,
    extra: int,
) -> np.ndarray:
    """Greedy marginal purchase. Returns one level per channel."""
    n = int(d2.shape[0])
    level = np.zeros(n, dtype=np.int64)
    spent = 0
    hi = np.maximum(np.asarray(d2, dtype=np.float64), 0.0)
    mid = np.minimum(np.maximum(np.asarray(d4, dtype=np.float64), 0.0), hi)
    lo = np.minimum(np.maximum(np.asarray(d8, dtype=np.float64), 0.0), mid)
    steps = ((2, 0, hi, mid), (4, 1, mid, lo), (8, 2, lo, None))
    while True:
        best_i = -1
        best_g = 0.0
        best_cost = 0
        for cost, src, upper, lower in steps:
            drop = upper if lower is None else upper - lower
            for i in range(n):
                if level[i] != src or spent + cost > extra or drop[i] <= 0:
                    continue
                gain = float(drop[i]) / cost
                if gain > best_g:
                    best_g = gain
                    best_i = i
                    best_cost = cost
        if best_i < 0:
            break
        level[best_i] += 1
        spent += best_cost
    return level


def allocate_levels(d2: np.ndarray, d4: np.ndarray, d8: np.ndarray, extra: int) -> np.ndarray:
    """``d*`` is ``[n_layers, n_kv, head_dim]``. Returns levels of the same shape."""
    levels = np.zeros(d2.shape, dtype=np.int64)
    n_layers, n_kv, _ = d2.shape
    for layer in range(n_layers):
        for head in range(n_kv):
            levels[layer, head] = waterfill(
                d2[layer, head], d4[layer, head], d8[layer, head], extra
            )
    return levels


def level_histogram(levels: np.ndarray) -> dict[str, float]:
    flat = np.asarray(levels).reshape(-1)
    n_heads = int(np.prod(np.asarray(levels).shape[:-1]))
    return {
        "fp16": float((flat == 3).sum() / n_heads),
        "int8": float((flat == 2).sum() / n_heads),
        "int4": float((flat == 1).sum() / n_heads),
        "int2": float((flat == 0).sum() / n_heads),
        "avg_bit": float(LEVEL_BITS[flat].mean()) if flat.size else float("nan"),
    }


def self_check() -> None:
    d2 = np.array([10.0, 8.0, 4.0])
    d4 = np.array([6.0, 1.0, 3.5])
    d8 = np.array([5.0, 0.2, 3.0])
    level = waterfill(d2, d4, d8, extra=6)
    if not np.array_equal(level, np.array([1, 1, 1])):
        raise RuntimeError(f"waterfill 6-bit got {level}")
    level2 = waterfill(d2, d4, d8, extra=10)
    if not np.array_equal(level2, np.array([2, 1, 1])):
        raise RuntimeError(f"waterfill 10-bit got {level2}")
    if extra_bits(2.484375) != 62:
        raise RuntimeError(f"extra bits got {extra_bits(2.484375)}")
    print("allocate self-check ok", flush=True)
