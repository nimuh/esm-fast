"""Parity of the from-scratch LayerNorm against torch.nn.LayerNorm."""

from __future__ import annotations

import torch

from esm_fast.modules.layer_norm import LayerNorm
from tests.conftest import ATOL, RTOL


def test_layer_norm_forward_matches_torch():
    dim = 128
    ours = LayerNorm(dim)
    ref = torch.nn.LayerNorm(dim)
    # Random (non-trivial) affine params so the test exercises weight/bias.
    with torch.no_grad():
        w = torch.randn(dim)
        b = torch.randn(dim)
        ours.weight.copy_(w)
        ours.bias.copy_(b)
        ref.weight.copy_(w)
        ref.bias.copy_(b)

    x = torch.randn(4, 16, dim)
    torch.testing.assert_close(ours(x), ref(x), atol=ATOL, rtol=RTOL)


def test_layer_norm_backward_matches_torch():
    dim = 64
    ours = LayerNorm(dim)
    ref = torch.nn.LayerNorm(dim)

    x1 = torch.randn(8, dim, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    ours(x1).sum().backward()
    ref(x2).sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, atol=ATOL, rtol=RTOL)
