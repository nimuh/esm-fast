"""Parity of the Triton tiled GEMM against ``torch.matmul``.

The kernel in ``esm_fast.kernels.matmul`` is still a stub; this file pins the
contract it has to satisfy:

    matmul(a, b) -> Tensor      # a: (M, K), b: (K, N) -> (M, N)

Both operands are float32 CUDA tensors. Shapes that are not multiples of the
block size are included on purpose so the boundary masking gets exercised.

Tolerances are looser than the repo-wide ``ATOL``/``RTOL``: ``tl.dot``
accumulates in TF32 by default, which costs about a dozen mantissa bits. TF32 is
switched off for the *reference* so the kernel is always compared against the
full-precision result rather than another approximation.

Skipped automatically on CPU-only machines (no CUDA / no Triton).
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import requires_triton

pytestmark = [pytest.mark.gpu, requires_triton]

# TF32 in ``tl.dot`` gives ~1e-3 relative error; keep the bound above that but
# tight enough that a wrong reduction order or a bad mask still fails.
MM_ATOL = 1e-2
MM_RTOL = 1e-2


@pytest.fixture(autouse=True)
def _reference_in_full_precision():
    """Force the torch reference to fp32 (no TF32) for the duration of a test."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev


@pytest.mark.parametrize(
    "m,k,n",
    [
        (128, 128, 128),  # square, block-aligned
        (256, 512, 128),  # long reduction dim
        (64, 32, 256),  # wide output
        (37, 61, 53),  # nothing divides evenly -> masking on all three axes
        (1, 64, 1),  # degenerate vector-vector case
    ],
)
def test_triton_matmul_matches_torch(m, k, n):
    from esm_fast.kernels.matmul import matmul

    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")

    # assert_close also pins dtype, shape and device of the kernel's output.
    torch.testing.assert_close(matmul(a, b), a @ b, atol=MM_ATOL, rtol=MM_RTOL)


def test_triton_matmul_handles_transposed_operand():
    """A transposed view is non-contiguous; the kernel must honour both strides."""
    from esm_fast.kernels.matmul import matmul

    a = torch.randn(128, 64, device="cuda")
    b = torch.randn(96, 64, device="cuda").t()  # (64, 96), stride (1, 64)

    torch.testing.assert_close(matmul(a, b), a @ b, atol=MM_ATOL, rtol=MM_RTOL)


def test_triton_matmul_is_deterministic():
    """Same inputs, same result -- no accumulation-order nondeterminism."""
    from esm_fast.kernels.matmul import matmul

    a = torch.randn(128, 256, device="cuda")
    b = torch.randn(256, 128, device="cuda")

    torch.testing.assert_close(matmul(a, b), matmul(a, b), atol=0.0, rtol=0.0)
