"""Load the packed decode kernel on demand.

Importing LogitFill does not compile CUDA. The first packed decode on a CUDA
tensor attempts compilation and falls back to the dense reference if the
toolchain is unavailable.
"""
from __future__ import annotations

from pathlib import Path

_MODULE = None


def get_kernel():
    """Return the compiled extension, or None when it cannot be built."""
    global _MODULE
    if _MODULE is False:
        return None
    if _MODULE is not None:
        return _MODULE
    try:
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_name("kernel.cu")
        _MODULE = load(
            name="logitfill_decode",
            sources=[str(source)],
            extra_cuda_cflags=["-O3"],
            verbose=False,
            with_cuda=True,
        )
    except Exception:
        _MODULE = False
        return None
    return _MODULE
