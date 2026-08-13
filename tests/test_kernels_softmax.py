"""Parity of the Triton fused softmax against torch.softmax.

Skipped automatically on CPU-only machines (no CUDA / no Triton).
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import requires_triton

pytestmark = [pytest.mark.gpu, requires_triton]


@pytest.mark.parametrize("shape", [(128, 64), (200, 1000), (1, 17)])
def test_triton_softmax_matches_torch(shape):
    from esm_fast.kernels.softmax import softmax

    x = torch.randn(*shape, device="cuda")
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-4)
