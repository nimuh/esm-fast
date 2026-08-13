"""Parity of the fused bias-add + GELU kernel against ``F.gelu(x + bias)``.

The kernel in ``esm_fast.kernels.fused_bias_gelu`` is still a stub; this file
pins the contract it has to satisfy:

    fused_bias_gelu(x, bias) -> Tensor   # x: (..., n), bias: (n,)

This fuses the first half of :class:`esm_fast.modules.feed_forward.FeedForward`
(``activation(fc1(x))``), so it must use the **exact erf** formulation -- that is
the default of :func:`torch.nn.functional.gelu` and of
:func:`esm_fast.functional.gelu`, and it is what ESM-2 uses. A tanh-approximate
kernel will (correctly) fail these tolerances.

Skipped automatically on CPU-only machines (no CUDA / no Triton).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from esm_fast.functional import gelu
from tests.conftest import ATOL, RTOL, requires_triton

pytestmark = [pytest.mark.gpu, requires_triton]


@pytest.mark.parametrize("shape", [(128, 64), (64, 512), (7, 33)])
def test_fused_bias_gelu_matches_torch(shape):
    from esm_fast.kernels.fused_bias_gelu import fused_bias_gelu

    n_cols = shape[-1]
    x = torch.randn(*shape, device="cuda")
    bias = torch.randn(n_cols, device="cuda")

    out = fused_bias_gelu(x, bias)

    torch.testing.assert_close(out, F.gelu(x + bias), atol=ATOL, rtol=RTOL)
    # ...and against the repo's own reference numerics, which the modules use.
    torch.testing.assert_close(out, gelu(x + bias), atol=ATOL, rtol=RTOL)


def test_fused_bias_gelu_is_exact_not_tanh_approximation():
    """Pin the erf formulation: the two variants differ well above tolerance."""
    from esm_fast.kernels.fused_bias_gelu import fused_bias_gelu

    x = torch.randn(256, 128, device="cuda")
    bias = torch.randn(128, device="cuda")

    exact = F.gelu(x + bias, approximate="none")
    approx = F.gelu(x + bias, approximate="tanh")
    # Guard the guard: if these ever agree to tolerance the test proves nothing.
    assert (exact - approx).abs().max() > 10 * ATOL

    torch.testing.assert_close(fused_bias_gelu(x, bias), exact, atol=ATOL, rtol=RTOL)


def test_fused_bias_gelu_handles_saturating_inputs():
    """Large |x| drives GELU to its asymptotes; no NaN/Inf from erf or exp."""
    from esm_fast.kernels.fused_bias_gelu import fused_bias_gelu

    x = torch.tensor(
        [[-30.0, -8.0, -1.0, 0.0, 1.0, 8.0, 30.0]],
        device="cuda",
    )
    bias = torch.zeros(7, device="cuda")

    out = fused_bias_gelu(x, bias)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, F.gelu(x), atol=ATOL, rtol=RTOL)


def test_fused_bias_gelu_matches_feed_forward_first_half():
    """End-to-end tie-in: the kernel can stand in for ``activation(fc1(x))``."""
    from esm_fast.kernels.fused_bias_gelu import fused_bias_gelu
    from esm_fast.modules.feed_forward import FeedForward

    ff = FeedForward(dim=64, hidden_dim=256).to("cuda")
    x = torch.randn(32, 64, device="cuda")

    fused = fused_bias_gelu(x @ ff.fc1.weight.t(), ff.fc1.bias)

    torch.testing.assert_close(fused, ff.activation(ff.fc1(x)), atol=ATOL, rtol=RTOL)
