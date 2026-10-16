"""Key quantization, attention-logit distortion, and reverse water-filling."""

from logitfill.quantization.allocate import (
    allocate_levels,
    assert_history_budget,
    extra_bits,
    kv_history_budget,
    waterfill,
)
from logitfill.quantization.distort import channel_curves, mean_curves, normalize_channel_curves
from logitfill.quantization.memory import runtime_footprint
from logitfill.quantization.quantizer import GroupQuantizer, make_quantizers, make_value_quantizer

__all__ = [
    "GroupQuantizer",
    "allocate_levels",
    "assert_history_budget",
    "channel_curves",
    "extra_bits",
    "kv_history_budget",
    "make_quantizers",
    "make_value_quantizer",
    "mean_curves",
    "normalize_channel_curves",
    "runtime_footprint",
    "waterfill",
]
