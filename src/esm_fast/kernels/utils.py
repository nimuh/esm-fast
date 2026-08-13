"""Guards for optional Triton/CUDA availability."""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def triton_available() -> bool:
    """True only when both Triton is importable and a CUDA device is present."""
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()
