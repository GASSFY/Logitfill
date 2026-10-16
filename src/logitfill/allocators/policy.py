"""Model-level Key precision policies.

A policy is one ``[n_layers, n_kv, head_dim]`` level map shared by every
request and every history page. Online inference loads it and does not run
the allocator again.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from logitfill import __version__
from logitfill.allocators.allocate import (
    LEVEL_BITS,
    achieved_history_bits,
    assert_history_budget,
    kv_history_budget,
    level_histogram,
)

POLICY_VERSION = 1
_LEVEL_NAMES = {0: "int2", 1: "int4", 2: "int8", 3: "fp16"}


class PrecisionPolicy:
    """Fixed per-channel Key levels plus the calibration metadata."""

    def __init__(self, levels: np.ndarray, metadata: dict, curves: dict | None = None):
        self.levels = np.asarray(levels, dtype=np.int64)
        if self.levels.ndim != 3:
            raise ValueError(f"levels must be [layers, kv_heads, head_dim], got {self.levels.shape}")
        if np.any((self.levels < 0) | (self.levels > 3)):
            raise ValueError("levels must be in {0, 1, 2, 3}")
        self.metadata = dict(metadata)
        self.curves = {key: np.asarray(value) for key, value in (curves or {}).items()}
        self._check_consistency()

    def _check_consistency(self) -> None:
        shape = self.levels.shape
        for key in ("n_layers", "n_kv", "head_dim"):
            if key in self.metadata and int(self.metadata[key]) != shape[{"n_layers": 0, "n_kv": 1, "head_dim": 2}[key]]:
                raise ValueError(f"metadata {key}={self.metadata[key]} does not match levels {shape}")
        if self.metadata.get("enforce_budget", True) and "target_kv_bit" in self.metadata:
            v_bits = int(self.metadata.get("v_bits", 2))
            assert_history_budget(self.levels, float(self.metadata["target_kv_bit"]), v_bits)
        for name, curve in self.curves.items():
            if curve.shape != self.levels.shape:
                raise ValueError(f"curve {name} has shape {curve.shape}, levels are {self.levels.shape}")

    def validate_model(self, model) -> None:
        """Raise when the loaded model cannot consume this policy."""
        config = model.config
        n_layers = int(config.num_hidden_layers)
        n_kv = int(config.num_key_value_heads)
        n_q = int(config.num_attention_heads)
        head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // n_q)
        expected = (n_layers, n_kv, head_dim)
        if tuple(self.levels.shape) != expected:
            raise ValueError(f"policy shape {self.levels.shape} does not match model {expected}")
        model_id = getattr(config, "name_or_path", None) or getattr(config, "_name_or_path", None)
        saved = self.metadata.get("model_id")
        if saved and model_id and Path(str(saved)).name != Path(str(model_id)).name and str(saved) != str(model_id):
            raise ValueError(f"policy model_id {saved} does not match loaded model {model_id}")

    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {"levels": self.levels}
        payload.update(self.curves)
        np.savez_compressed(path / "policy.npz", **payload)
        (path / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "PrecisionPolicy":
        path = Path(directory)
        packed = np.load(path / "policy.npz")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        curves = {key: packed[key] for key in packed.files if key != "levels"}
        return cls(packed["levels"], metadata, curves)


def build_policy(
    levels: np.ndarray,
    *,
    model_id: str,
    target_kv_bit: float,
    v_bits: int = 2,
    k_group_size: int = 128,
    v_group_size: int = 128,
    page_size: int = 128,
    calibration_datasets: list[str] | None = None,
    n_samples: int = 0,
    curves: dict | None = None,
    normalized: bool = True,
) -> PrecisionPolicy:
    """Assemble a checked policy from an allocator output."""
    values = np.asarray(levels, dtype=np.int64)
    budget = assert_history_budget(values, target_kv_bit, v_bits)
    achieved = achieved_history_bits(values, v_bits)
    metadata = {
        "policy_version": POLICY_VERSION,
        "code_version": __version__,
        "model_id": model_id,
        "n_layers": int(values.shape[0]),
        "n_kv": int(values.shape[1]),
        "head_dim": int(values.shape[2]),
        "k_group_size": int(k_group_size),
        "v_group_size": int(v_group_size),
        "page_size": int(page_size),
        "v_bits": int(v_bits),
        "target_kv_bit": float(target_kv_bit),
        "target_k_bit": budget["target_k_bit"],
        "extra_bits_per_head": budget["extra_bits_per_head"],
        "calibration_datasets": list(calibration_datasets or []),
        "n_samples": int(n_samples),
        "normalized": bool(normalized),
        "histogram": level_histogram(values),
        "achieved": achieved,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return PrecisionPolicy(values, metadata, curves)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        stop = start + 1
        while stop < values.shape[0] and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            average = 0.5 * (start + stop - 1)
            ranks[order[start:stop]] = average
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rank correlation. NaN when either side has no variation."""
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"spearman inputs differ: {x.shape} vs {y.shape}")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.square(rx).sum() * np.square(ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def overlap_report(left: PrecisionPolicy, right: PrecisionPolicy) -> dict:
    """Agreement of two policies. Meaningful when model, shape, and budget match."""
    if left.levels.shape != right.levels.shape:
        raise ValueError(
            f"policy shapes differ: {left.levels.shape} vs {right.levels.shape}. "
            "Overlap is only defined for the same model layout."
        )
    same_budget = (
        left.metadata.get("target_kv_bit") == right.metadata.get("target_kv_bit")
        and int(left.metadata.get("v_bits", 2)) == int(right.metadata.get("v_bits", 2))
    )
    report = {
        "same_shape": True,
        "same_budget": bool(same_budget),
        "exact_agreement": float(np.mean(left.levels == right.levels)),
        "avg_bit_left": float(LEVEL_BITS[left.levels].mean()),
        "avg_bit_right": float(LEVEL_BITS[right.levels].mean()),
    }
    report["avg_bit_abs_diff"] = abs(report["avg_bit_left"] - report["avg_bit_right"])
    for code, name in _LEVEL_NAMES.items():
        report[f"jaccard_{name}"] = _jaccard(left.levels == code, right.levels == code)
    if "d2" in left.curves and "d4" in left.curves and "d2" in right.curves and "d4" in right.curves:
        gain_left = (left.curves["d2"] - left.curves["d4"]) / 2.0
        gain_right = (right.curves["d2"] - right.curves["d4"]) / 2.0
        correlation = spearman(gain_left, gain_right)
        report["spearman_gain_2to4"] = None if correlation != correlation else correlation
    else:
        report["spearman_gain_2to4"] = None
    if not same_budget:
        report["warning"] = (
            "budgets differ; agreement does not measure model-structure stability"
        )
    return report


def consensus_levels(policies: list[PrecisionPolicy]) -> np.ndarray:
    """Per-channel majority level. Ties keep the lower-bit choice."""
    if not policies:
        raise ValueError("consensus requires at least one policy")
    shape = policies[0].levels.shape
    if any(policy.levels.shape != shape for policy in policies):
        raise ValueError("consensus requires policies with identical shapes")
    stacked = np.stack([policy.levels for policy in policies], axis=0)
    flat = stacked.reshape(stacked.shape[0], -1)
    chosen = np.empty(flat.shape[1], dtype=np.int64)
    for index in range(flat.shape[1]):
        values, counts = np.unique(flat[:, index], return_counts=True)
        chosen[index] = int(values[counts == counts.max()].min())
    return chosen.reshape(shape)
