"""Key-channel allocation and the saved level map."""

from logitfill.allocators.allocate import (
    allocate_levels,
    assert_history_budget,
    extra_bits,
    kv_history_budget,
    waterfill,
)
from logitfill.allocators.memory import runtime_footprint
from logitfill.allocators.policy import PrecisionPolicy, build_policy

__all__ = [
    "PrecisionPolicy",
    "allocate_levels",
    "assert_history_budget",
    "build_policy",
    "extra_bits",
    "kv_history_budget",
    "runtime_footprint",
    "waterfill",
]
