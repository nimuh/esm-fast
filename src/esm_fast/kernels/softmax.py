"""Fused row-wise softmax in Triton.

Computes ``softmax(x, dim=-1)`` for a 2D tensor, one program per row, using the
numerically-stable max-subtraction trick. Validated against
:func:`torch.softmax` in ``tests/test_kernels_softmax.py``.

Imports of ``triton`` are deferred so this module (and its enclosing package)
remain importable on CPU-only machines; only calling :func:`softmax` requires a
GPU.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _build():
    import triton
    import triton.language as tl

    @triton.jit
    def _softmax_kernel(
        x_ptr,
        out_ptr,
        x_row_stride,
        out_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        x = tl.load(x_ptr + row * x_row_stride + col, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        out = num / tl.sum(num, axis=0)
        tl.store(out_ptr + row * out_row_stride + col, out, mask=mask)

    return _softmax_kernel


_KERNEL = None


def softmax(x: Tensor) -> Tensor:
    """Row-wise softmax over the last dim of a 2D CUDA tensor."""
    global _KERNEL
    import triton

    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")
    if not x.is_cuda:
        raise RuntimeError("triton softmax requires a CUDA tensor")

    if _KERNEL is None:
        _KERNEL = _build()

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)
    _KERNEL[(n_rows,)](
        x,
        out,
        x.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return out
