"""Parity of the Triton LayerNorm backward pass against autograd.

The kernel in ``esm_fast.kernels.layer_norm_bwd`` is still a stub; this file
pins the contract it has to satisfy:

    layer_norm_bwd(dy, x, weight, eps=1e-5) -> (dx, dweight, dbias)

``x`` and ``dy`` are 2D float32 CUDA tensors of the same shape, ``weight`` is
``(n_cols,)``. The bias value is not needed for the backward pass (``dbias`` is
just the column sum of ``dy``), so it is not part of the signature.

The reference is autograd through :func:`torch.nn.functional.layer_norm`, which
is what the forward kernel in ``esm_fast.kernels.layer_norm`` is already pinned
to.

Skipped automatically on CPU-only machines (no CUDA / no Triton).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import ATOL, RTOL, requires_triton

pytestmark = [pytest.mark.gpu, requires_triton]

# dweight/dbias reduce over every row, so they accumulate more error than dx --
# and the kernel is free to reduce in a different order than autograd does.
REDUCTION_ATOL = 1e-4
REDUCTION_RTOL = 1e-3


def _torch_reference(shape, eps):
    """Grads of ``F.layer_norm`` for a random problem, plus the inputs used."""
    n_cols = shape[1]
    x = torch.randn(*shape, device="cuda", requires_grad=True)
    weight = torch.randn(n_cols, device="cuda", requires_grad=True)
    bias = torch.randn(n_cols, device="cuda", requires_grad=True)
    dy = torch.randn(*shape, device="cuda")

    F.layer_norm(x, (n_cols,), weight, bias, eps=eps).backward(dy)

    inputs = (dy, x.detach(), weight.detach())
    grads = (x.grad, weight.grad, bias.grad)
    return inputs, grads


@pytest.mark.parametrize("shape", [(128, 64), (64, 512), (7, 33)])
def test_triton_layer_norm_bwd_matches_autograd(shape):
    from esm_fast.kernels.layer_norm_bwd import layer_norm_bwd

    eps = 1e-5
    (dy, x, weight), (dx_ref, dw_ref, db_ref) = _torch_reference(shape, eps)

    dx, dweight, dbias = layer_norm_bwd(dy, x, weight, eps=eps)

    torch.testing.assert_close(dx, dx_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(dweight, dw_ref, atol=REDUCTION_ATOL, rtol=REDUCTION_RTOL)
    torch.testing.assert_close(dbias, db_ref, atol=REDUCTION_ATOL, rtol=REDUCTION_RTOL)


def test_triton_layer_norm_bwd_respects_eps():
    """A different eps has to change dx -- it must not be hard-coded."""
    from esm_fast.kernels.layer_norm_bwd import layer_norm_bwd

    eps = 1e-2
    (dy, x, weight), (dx_ref, _, _) = _torch_reference((64, 128), eps)

    dx, _, _ = layer_norm_bwd(dy, x, weight, eps=eps)

    torch.testing.assert_close(dx, dx_ref, atol=ATOL, rtol=RTOL)


def test_triton_layer_norm_bwd_dbias_is_column_sum():
    """Sanity check on the cheapest of the three grads, independent of autograd."""
    from esm_fast.kernels.layer_norm_bwd import layer_norm_bwd

    x = torch.randn(256, 64, device="cuda")
    weight = torch.randn(64, device="cuda")
    dy = torch.randn(256, 64, device="cuda")

    _, _, dbias = layer_norm_bwd(dy, x, weight, eps=1e-5)

    torch.testing.assert_close(dbias, dy.sum(dim=0), atol=REDUCTION_ATOL, rtol=REDUCTION_RTOL)


def test_triton_layer_norm_bwd_matches_forward_kernel_gradient():
    """The backward kernel must differentiate *our* forward kernel, not a variant.

    Runs the Triton forward, then checks that the Triton backward reproduces the
    gradient autograd computes for a numerically identical eager forward.
    """
    from esm_fast.kernels.layer_norm import layer_norm
    from esm_fast.kernels.layer_norm_bwd import layer_norm_bwd

    eps = 1e-5
    x = torch.randn(96, 128, device="cuda")
    weight = torch.randn(128, device="cuda")
    bias = torch.randn(128, device="cuda")
    dy = torch.randn(96, 128, device="cuda")

    x_ref = x.clone().requires_grad_(True)
    out_ref = F.layer_norm(x_ref, (128,), weight, bias, eps=eps)
    out_ref.backward(dy)

    torch.testing.assert_close(layer_norm(x, weight, bias, eps=eps), out_ref, atol=ATOL, rtol=RTOL)

    dx, _, _ = layer_norm_bwd(dy, x, weight, eps=eps)
    torch.testing.assert_close(dx, x_ref.grad, atol=ATOL, rtol=RTOL)
