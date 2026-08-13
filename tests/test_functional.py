"""Parity of the functional core against torch.nn.functional."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from esm_fast.functional import gelu, scaled_dot_product_attention
from tests.conftest import ATOL, RTOL


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_matches_torch(approximate):
    x = torch.randn(64, 128)
    torch.testing.assert_close(
        gelu(x, approximate=approximate),
        F.gelu(x, approximate=approximate),
        atol=ATOL,
        rtol=RTOL,
    )


def test_sdpa_matches_torch_unmasked():
    q = torch.randn(2, 8, 16, 32)
    k = torch.randn(2, 8, 16, 32)
    v = torch.randn(2, 8, 16, 32)
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
        atol=ATOL,
        rtol=RTOL,
    )


def test_sdpa_matches_torch_causal():
    q = torch.randn(2, 4, 10, 16)
    k = torch.randn(2, 4, 10, 16)
    v = torch.randn(2, 4, 10, 16)
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v, is_causal=True),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
        atol=ATOL,
        rtol=RTOL,
    )


def test_sdpa_matches_torch_bool_mask():
    q = torch.randn(2, 4, 10, 16)
    k = torch.randn(2, 4, 10, 16)
    v = torch.randn(2, 4, 10, 16)
    mask = torch.randint(0, 2, (10, 10), dtype=torch.bool)
    # Guarantee at least one attended key per row to avoid all -inf rows.
    mask[:, 0] = True
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v, attn_mask=mask),
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
        atol=ATOL,
        rtol=RTOL,
    )


def test_sdpa_rejects_mask_and_causal():
    q = k = v = torch.randn(1, 1, 4, 8)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(4, 4, dtype=torch.bool), is_causal=True)
