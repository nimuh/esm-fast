"""Parity of the Triton fused LayerNorm against F.layer_norm.

Skipped automatically on CPU-only machines (no CUDA / no Triton).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import requires_triton

pytestmark = [pytest.mark.gpu, requires_triton]


@pytest.mark.parametrize("shape", [(128, 64), (64, 512), (7, 33)])
def test_triton_layer_norm_matches_torch(shape):
    from esm_fast.kernels.layer_norm import layer_norm

    n_cols = shape[1]
    x = torch.randn(*shape, device="cuda")
    weight = torch.randn(n_cols, device="cuda")
    bias = torch.randn(n_cols, device="cuda")

    torch.testing.assert_close(
        layer_norm(x, weight, bias, eps=1e-5),
        F.layer_norm(x, (n_cols,), weight, bias, eps=1e-5),
        atol=1e-5,
        rtol=1e-4,
    )
