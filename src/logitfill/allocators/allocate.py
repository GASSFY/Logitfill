"""Reverse water-filling for Key channels.

Levels are 0=int2, 1=int4, 2=int8, 3=fp16. Each head starts at int2.
Extra bits are counted above that floor. One purchase upgrades a channel
by a single step: 2→4 costs 2, 4→8 costs 4, 8→16 costs 8.
"""
from __future__ import annotations

import numpy as np

LEVEL_BITS = np.array([2, 4, 8, 16], dtype=np.int64)


def extra_budget_per_head(target_bit_k: float, n_channels: int = 128) -> int:
    """Integer Key bits per head above the int2 floor. This is not ``target_bit``."""
    spare = (float(target_bit_k) - 2.0) * n_channels
    if spare < 0:
        raise ValueError(f"target_bit_k {target_bit_k} is below the int2 floor")
    return int(round(spare))


# Extra payload bits above the int2 floor for one channel at each level.
LEVEL_EXTRA = np.array([0, 2, 6, 14], dtype=np.int64)


def kv_history_budget(target_bit: float, bit_v: int = 2, n_channels: int = 128) -> dict:
    """Convert a K+V history payload budget into a per-head Key budget.

    ``target_bit`` is the average of Key and Value code bits on quantized
    history tokens. The Value width is fixed, so the Key target is

        target_bit_k = 2 * target_bit - bit_v.

    Residual BF16 pages and scale/min metadata are not part of this budget.
    ``extra_budget_per_head`` rounds to an integer per head, so the achieved rate can
    differ from the request by less than half a bit per head.
    """
    if int(bit_v) < 2 or int(bit_v) > 16:
        raise ValueError(f"bit_v must be in 2..16, got {bit_v}")
    target_k = 2.0 * float(target_bit) - float(bit_v)
    if target_k < 2.0 or target_k > 16.0:
        raise ValueError(
            f"target_bit {target_bit} with bit_v {bit_v} implies "
            f"Key target {target_k}, outside [2, 16]"
        )
    extra = extra_budget_per_head(target_k, n_channels)
    return {
        "target_bit": float(target_bit),
        "bit_v": int(bit_v),
        "target_bit_k": float(target_k),
        "extra_budget_per_head": int(extra),
        "n_channels": int(n_channels),
    }


def spent_extra_per_head(levels: np.ndarray) -> np.ndarray:
    """Extra Key bits above int2. ``levels`` is ``[..., head_dim]``."""
    values = np.asarray(levels, dtype=np.int64)
    if np.any((values < 0) | (values > 3)):
        raise ValueError("levels must be in {0, 1, 2, 3}")
    return LEVEL_EXTRA[values].sum(axis=-1)


def achieved_history_bits(levels: np.ndarray, bit_v: int = 2) -> dict[str, float]:
    """Mean Key / Value / joint payload bits implied by a level map."""
    values = np.asarray(levels, dtype=np.int64)
    actual_bit_k = float(LEVEL_BITS[values].mean()) if values.size else float("nan")
    return {
        "actual_bit_k": actual_bit_k,
        "bit_v": float(bit_v),
        "actual_bit_kv": 0.5 * (actual_bit_k + float(bit_v)),
    }


def assert_history_budget(levels: np.ndarray, target_bit: float, bit_v: int = 2) -> dict:
    """Reject a level map that spends more than the per-head Key budget."""
    values = np.asarray(levels, dtype=np.int64)
    budget = kv_history_budget(target_bit, bit_v, int(values.shape[-1]))
    spent = spent_extra_per_head(values)
    overflow = int(spent.max()) if spent.size else 0
    if overflow > budget["extra_budget_per_head"]:
        raise ValueError(
            f"waterfill spent {overflow} extra bits in a head, "
            f"budget is {budget['extra_budget_per_head']}"
        )
    budget["extra_spent_per_head"] = overflow
    budget.update(achieved_history_bits(values, bit_v))
    return budget


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
        "actual_bit_k": float(LEVEL_BITS[flat].mean()) if flat.size else float("nan"),
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
    if extra_budget_per_head(2.484375) != 62:
        raise RuntimeError(f"extra bits got {extra_budget_per_head(2.484375)}")
    budget = kv_history_budget(2.5, bit_v=2, n_channels=128)
    if abs(budget["target_bit_k"] - 3.0) > 1e-12 or budget["extra_budget_per_head"] != 128:
        raise RuntimeError(f"kv budget got {budget}")
    checked = assert_history_budget(level, target_bit=3.0, bit_v=2)
    if checked["extra_spent_per_head"] != 6:
        raise RuntimeError(f"spent extra got {checked}")
    print("allocate self-check ok", flush=True)
