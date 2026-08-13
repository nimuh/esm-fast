"""Parity of the FlashAttention-style fused kernel against PyTorch SDPA.

The kernel in ``esm_fast.kernels.fused_attention`` is still a stub; this file
pins the contract it has to satisfy:

    fused_attention(q, k, v, causal=False, scale=None) -> Tensor

with ``q``/``k``/``v`` of shape ``(batch, heads, seq, head_dim)`` (float32,
CUDA) and ``scale`` defaulting to ``1 / sqrt(head_dim)`` -- the same defaults as
:func:`torch.nn.functional.scaled_dot_product_attention` and as
:func:`esm_fast.functional.scaled_dot_product_attention`, both of which are used
as references here.

Tolerances are looser than the repo-wide ``ATOL``/``RTOL``: the two ``tl.dot``
calls in a flash-attention inner loop accumulate in TF32 by default, and the
online-softmax rescaling makes the summation order differ from the eager
reference regardless. TF32 is disabled for the reference so the kernel is always
compared against the full-precision result.

Skipped automatically on CPU-only machines (no CUDA / no Triton).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from esm_fast.functional import scaled_dot_product_attention as reference_sdpa
from tests.conftest import requires_triton

pytestmark = [pytest.mark.gpu, requires_triton]

# TF32 dots plus online softmax rescaling; ~1e-3 relative error is expected.
ATTN_ATOL = 1e-2
ATTN_RTOL = 1e-2


@pytest.fixture(autouse=True)
def _reference_in_full_precision():
    """Force the torch reference to fp32 (no TF32) for the duration of a test."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev


def _qkv(batch, heads, seq, head_dim, k_seq=None):
    shape = (batch, heads, seq, head_dim)
    kv_shape = (batch, heads, k_seq or seq, head_dim)
    return (
        torch.randn(*shape, device="cuda"),
        torch.randn(*kv_shape, device="cuda"),
        torch.randn(*kv_shape, device="cuda"),
    )


@pytest.mark.parametrize(
    "batch,heads,seq,head_dim",
    [
        (1, 1, 128, 64),
        (2, 8, 256, 64),
        (2, 4, 64, 32),
        (1, 2, 77, 32),  # sequence length not a multiple of any block size
    ],
)
def test_fused_attention_matches_torch_sdpa(batch, heads, seq, head_dim):
    from esm_fast.kernels.fused_attention import fused_attention

    q, k, v = _qkv(batch, heads, seq, head_dim)

    out = fused_attention(q, k, v)

    torch.testing.assert_close(
        out, F.scaled_dot_product_attention(q, k, v), atol=ATTN_ATOL, rtol=ATTN_RTOL
    )
    # ...and against the repo's own reference numerics.
    torch.testing.assert_close(out, reference_sdpa(q, k, v), atol=ATTN_ATOL, rtol=ATTN_RTOL)


@pytest.mark.parametrize("seq", [128, 77])
def test_fused_attention_causal_matches_torch_sdpa(seq):
    from esm_fast.kernels.fused_attention import fused_attention

    q, k, v = _qkv(2, 4, seq, 64)

    torch.testing.assert_close(
        fused_attention(q, k, v, causal=True),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
        atol=ATTN_ATOL,
        rtol=ATTN_RTOL,
    )


def test_fused_attention_causal_actually_masks():
    """Changing a future key must not change an earlier query's output.

    Guards against the causal flag being accepted but ignored -- which the parity
    check above can miss if the mask is only approximately right.
    """
    from esm_fast.kernels.fused_attention import fused_attention

    q, k, v = _qkv(1, 1, 64, 32)
    out = fused_attention(q, k, v, causal=True)

    v_perturbed = v.clone()
    v_perturbed[:, :, 32:] += 10.0
    out_perturbed = fused_attention(q, k, v_perturbed, causal=True)

    torch.testing.assert_close(
        out[:, :, :32], out_perturbed[:, :, :32], atol=ATTN_ATOL, rtol=ATTN_RTOL
    )
    assert not torch.allclose(out[:, :, 32:], out_perturbed[:, :, 32:], atol=ATTN_ATOL)


def test_fused_attention_honours_custom_scale():
    from esm_fast.kernels.fused_attention import fused_attention

    q, k, v = _qkv(2, 4, 128, 64)
    scale = 0.137

    torch.testing.assert_close(
        fused_attention(q, k, v, scale=scale),
        F.scaled_dot_product_attention(q, k, v, scale=scale),
        atol=ATTN_ATOL,
        rtol=ATTN_RTOL,
    )
    # The default must be 1/sqrt(head_dim), not something baked in.
    torch.testing.assert_close(
        fused_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(64)),
        atol=ATTN_ATOL,
        rtol=ATTN_RTOL,
    )


def test_fused_attention_supports_cross_attention_shapes():
    """Query and key/value sequence lengths need not match."""
    from esm_fast.kernels.fused_attention import fused_attention

    q, k, v = _qkv(2, 4, 64, 32, k_seq=96)

    torch.testing.assert_close(
        fused_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
        atol=ATTN_ATOL,
        rtol=ATTN_RTOL,
    )


def test_fused_attention_is_numerically_stable_with_large_logits():
    """Online softmax must subtract a running max -- no overflow to NaN/Inf."""
    from esm_fast.kernels.fused_attention import fused_attention

    q, k, v = _qkv(1, 2, 128, 64)
    q = q * 50.0  # logits well past exp() overflow without max subtraction

    out = fused_attention(q, k, v)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out, F.scaled_dot_product_attention(q, k, v), atol=ATTN_ATOL, rtol=ATTN_RTOL
    )
