"""Locations of the cloned baseline checkouts next to this repository."""
from __future__ import annotations

import sys
from pathlib import Path

_CHECKOUTS = {
    "pm_kvq": "PM-KVQ",
    "kivi": "KIVI",
    "kvtuner": "KVTuner/flexible_quant",
    "mixkvq": "MixKVQ",
}


def quantization_root() -> Path:
    return Path(__file__).resolve().parents[4]


def checkout(name: str) -> Path:
    path = quantization_root() / _CHECKOUTS[name]
    if not path.is_dir():
        raise FileNotFoundError(f"baseline checkout {name} is not at {path}")
    return path


def ensure_importable(name: str) -> Path:
    path = checkout(name)
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return path
