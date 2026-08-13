"""Shared test fixtures and helpers."""

from __future__ import annotations

import pytest
import torch

from esm_fast.kernels.utils import triton_available

# Deterministic tolerances for float32 parity checks against PyTorch.
ATOL = 1e-5
RTOL = 1e-4


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


requires_triton = pytest.mark.skipif(
    not triton_available(),
    reason="requires a CUDA device with Triton installed",
)
