"""Fused LayerNorm forward pass in Triton.

Computes ``layer_norm(x, weight, bias)`` over the last dimension of a 2D tensor,
one program per row. Validated against :func:`torch.nn.functional.layer_norm` in
``tests/test_kernels_layer_norm.py``.

As with the softmax kernel, ``triton`` is imported lazily so CPU-only imports
work; calling :func:`layer_norm` requires a GPU.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _build():
    import triton
    import triton.language as tl

    @triton.jit
    def _layer_norm_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        out_ptr,
        x_row_stride,
        out_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK_SIZE)
        mask = col < n_cols

        x = tl.load(x_ptr + row * x_row_stride + col, mask=mask, other=0.0)
        mean = tl.sum(x, axis=0) / n_cols
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)

        w = tl.load(w_ptr + col, mask=mask, other=0.0)
        b = tl.load(b_ptr + col, mask=mask, other=0.0)
        out = centered * rstd * w + b
        tl.store(out_ptr + row * out_row_stride + col, out, mask=mask)

    return _layer_norm_kernel


_KERNEL = None


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm over the last dim of a 2D CUDA tensor."""
    global _KERNEL
    import triton

    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")
    if not x.is_cuda:
        raise RuntimeError("triton layer_norm requires a CUDA tensor")

    if _KERNEL is None:
        _KERNEL = _build()

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)
    _KERNEL[(n_rows,)](
        x,
        weight,
        bias,
        out,
        x.stride(0),
        out.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    return out
